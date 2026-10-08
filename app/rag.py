"""Pipeline RAG: ingest -> retrieve -> (guard) -> generate -> sitasi."""
from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import settings
from .ingest import ParsedDoc, _load_catalog, parse_file, slugify
from .llm import LLM, LLMError
from .retrieval import Embedder, Hit, KnowledgeIndex, tokenize

log = logging.getLogger("rag.engine")

NOT_FOUND_TOKEN = "TIDAK_DITEMUKAN"

# System prompt mode RAG (ada konteks dokumen)
SYSTEM_PROMPT_RAG = f"""Anda adalah asisten AI SDM Polri yang cerdas dan membantu. \
Jawab pertanyaan pengguna berdasarkan KUTIPAN DOKUMEN yang diberikan pada bagian KONTEKS.

Aturan:
1. Utamakan informasi dari KONTEKS. Setiap pernyataan faktual dari dokumen diberi rujukan nomor kutipan, contoh [1] atau [2][3].
2. Jika KONTEKS tidak memuat jawaban, jawab: {NOT_FOUND_TOKEN}
3. Jangan mencampur isi dokumen yang berbeda jika tidak relevan.
4. Jawab dalam Bahasa Indonesia yang jelas dan ringkas. Gunakan daftar bernomor atau "-" untuk beberapa poin.
5. Pertahankan angka, nama pangkat, tanggal, dan istilah persis seperti di dokumen."""

# System prompt mode GENERAL (tidak ada konteks dokumen relevan)
SYSTEM_PROMPT_GENERAL = """Anda adalah asisten AI SDM Polri yang cerdas, ramah, dan membantu. \
Anda dapat menjawab pertanyaan umum, percakapan sehari-hari, maupun topik di luar dokumen knowledge base.

Aturan:
1. Jawab dengan ramah, helpful, dan natural dalam Bahasa Indonesia.
2. Jika pertanyaan berkaitan dengan kebijakan SDM Polri atau dokumen internal Polri, \
sarankan pengguna untuk menanyakan hal spesifik agar bisa dicari di knowledge base.
3. Jangan berpura-pura memiliki akses ke data internal Polri jika tidak ada di konteks.
4. Jawab langsung ke inti pertanyaan. Jangan memulai dengan sapaan (\"Halo\", \"Selamat pagi\", dll) kecuali pengguna menyapa lebih dulu."""

NOT_FOUND_ANSWER = (
    "Maaf, informasi tersebut **tidak ditemukan** pada knowledge base yang tersedia"
    "{docs}. Coba ubah kata kunci pertanyaan, atau tambahkan dokumen yang memuat informasinya."
)



@dataclass
class Source:
    n: int
    doc_id: str
    doc_title: str
    doc_short: str
    page_start: int
    page_end: int
    section: str
    snippet: str
    text: str
    relevance: float
    scores: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class Answer:
    answer: str
    found: bool
    sources: list[Source] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "found": self.found,
            "sources": [s.to_dict() for s in self.sources],
            "meta": self.meta,
        }


class RAGEngine:
    def __init__(self) -> None:
        self.llm = LLM()
        self.index = KnowledgeIndex()
        self.docs: dict[str, ParsedDoc] = {}
        self._lock = threading.RLock()
        self.ready = False
        self.error: str | None = None
        self.llm_error: str | None = None  # galat LLM terakhir (None = terakhir sukses)

    # ------------------------------------------------------------------ ingest
    def _aliases(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for fname, meta in _load_catalog().items():
            did = meta.get("id") or slugify(Path(fname).stem)
            if did in self.docs and meta.get("aliases"):
                out[did] = meta["aliases"]
        return out

    def rebuild(self) -> None:
        with self._lock:
            chunks = [c for d in self.docs.values() for c in d.chunks]
            new_index = KnowledgeIndex()
            new_index.embedder = self.index.embedder  # model dipakai ulang
            new_index.build(chunks, self._aliases())
            self.index = new_index

    def load_all(self) -> None:
        """Muat semua dokumen di DOCS_DIR (dipanggil saat startup)."""
        settings.DOCS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            for f in sorted(settings.DOCS_DIR.glob("*")):
                if f.suffix.lower() in {".pdf", ".txt", ".md"}:
                    try:
                        d = parse_file(f)
                        self.docs[d.doc_id] = d
                    except Exception as e:  # noqa: BLE001
                        log.error("Gagal memproses %s: %s", f.name, e)
            self.rebuild()
            self.ready = True
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
            log.exception("Gagal membangun index")
            return
        if settings.EMBEDDING_BACKEND != "none":
            threading.Thread(target=self._upgrade_dense, daemon=True, name="embedder").start()

    def _upgrade_dense(self) -> None:
        """Muat model embedding di background (butuh internet saat pertama kali). Gagal -> tetap BM25."""
        try:
            emb = Embedder()
            if not emb.ready:
                self.index.embedder = emb  # simpan pesan error untuk /api/health
                return
            self.index.embedder = emb
            self.rebuild()  # bangun vektor lalu tukar index secara atomik
            log.info("Retrieval di-upgrade ke hybrid (BM25 + embedding)")
        except Exception as e:  # noqa: BLE001
            log.warning("Upgrade embedding dilewati, tetap BM25 (%s: %s)", type(e).__name__, e)

    def _gen(self, system: str, user: str) -> str:
        """Panggil LLM sambil mencatat status terakhir (dipakai /api/health)."""
        try:
            out = self.llm.generate(system, user)
        except LLMError as e:
            self.llm_error = str(e)[:300]
            raise
        self.llm_error = None
        return out

    @staticmethod
    def _smalltalk(question: str) -> str:
        """Jawaban lokal untuk basa-basi saat LLM offline. Kosong bila bukan basa-basi."""
        q = re.sub(r"[^a-z0-9 ]", " ", question.lower()).strip()
        words = q.split()
        if not words or len(words) > 6:
            return ""
        hint = (
            "Saat ini mode AI sedang offline, tetapi pencarian dokumen tetap berfungsi. "
            "Silakan tanyakan hal spesifik, misalnya *persyaratan SBP T.A. 2027* atau "
            "*ketentuan penilaian kinerja anggota Polri*."
        )
        if words[0] in {"halo", "hai", "hallo", "hello", "hi", "p", "selamat", "assalamualaikum", "permisi"}:
            return f"Halo! Saya asisten informasi SDM Polri. {hint}"
        if "terima" in words or "makasih" in words or "thanks" in words:
            return "Sama-sama! Jika ada pertanyaan lain tentang dokumen SDM Polri, silakan tanyakan."
        if "siapa" in words and ("kamu" in words or "anda" in words):
            return f"Saya asisten informasi SDM Polri yang menjawab berdasarkan dokumen resmi beserta sumbernya. {hint}"
        return ""

    @staticmethod
    def _llm_down_message(err: str) -> str:
        if "kuota/kredit habis" in err or "no credits" in err or "insufficient_quota" in err:
            reason = "**kredit/kuota API habis** (HTTP 429) - tambah saldo di platform.openai.com/settings/organization/billing"
        elif "401" in err or "403" in err:
            reason = "**API key tidak valid atau sudah kedaluwarsa** (HTTP 401/403)"
        elif "429" in err:
            reason = "terlalu banyak permintaan / kuota API terlampaui (HTTP 429), coba lagi sebentar"
        elif "belum diisi" in err or "tidak dikonfigurasi" in err:
            reason = "API key belum diisi"
        else:
            reason = "layanan AI tidak dapat dihubungi (cek koneksi internet)"
        return (
            f"⚠️ **Mode AI sedang offline** - {reason}.\n\n"
            "**Cara memperbaiki:** isi `OPENAI_API_KEY` (atau `GEMINI_API_KEY`) yang valid di file `.env`, "
            "lalu jalankan `docker compose up -d --force-recreate`.\n\n"
            "Sementara itu pencarian dokumen tanpa AI tetap berfungsi - "
            "coba tanyakan hal spesifik tentang isi dokumen (mis. persyaratan SBP atau penilaian kinerja)."
        )
    def add_file(self, path: Path) -> ParsedDoc:
        doc = parse_file(path)
        if not doc.chunks:
            raise ValueError("Tidak ada teks yang dapat diekstrak dari dokumen (PDF scan memerlukan OCR).")
        with self._lock:
            # id unik
            base, i = doc.doc_id, 2
            while doc.doc_id in self.docs and self.docs[doc.doc_id].filename != path.name:
                doc.doc_id = f"{base}-{i}"
                i += 1
            for c in doc.chunks:
                c.doc_id = doc.doc_id
                c.id = c.id.replace(base, doc.doc_id, 1)
            self.docs[doc.doc_id] = doc
            self.rebuild()
        return doc

    def remove(self, doc_id: str) -> bool:
        with self._lock:
            doc = self.docs.pop(doc_id, None)
            if not doc:
                return False
            (settings.DOCS_DIR / doc.filename).unlink(missing_ok=True)
            self.rebuild()
            return True

    @staticmethod
    def _template(d: ParsedDoc) -> str:
        """Template chunking yang dipakai (ala RAGFlow): 'laws' = per Pasal/BAB, 'general' = jendela bertumpuk."""
        if d.chunks and sum(1 for c in d.chunks if "Pasal" in (c.section or "")) / len(d.chunks) >= 0.3:
            return "laws"
        return "general"

    def list_chunks(self, doc_id: str) -> list[dict] | None:
        d = self.docs.get(doc_id)
        if not d:
            return None
        return [
            {"n": i, "page_start": c.page_start, "page_end": c.page_end, "section": c.section,
             "chars": len(c.text), "text": c.text}
            for i, c in enumerate(d.chunks, 1)
        ]

    def list_docs(self) -> list[dict]:
        catalog = _load_catalog()
        return [
            {
                "id": d.doc_id,
                "title": d.title,
                "short": d.short,
                "filename": d.filename,
                "pages": len(d.pages),
                "chunks": len(d.chunks),
                "method": d.method,
                "template": self._template(d),
                "builtin": d.filename in catalog,
            }
            for d in self.docs.values()
        ]

    def status(self) -> dict:
        return {
            "ready": self.ready,
            "error": self.error,
            "documents": len(self.docs),
            "chunks": len(self.index.chunks),
            "retrieval": self.index.mode,
            "vector_store": self.index.store.backend if self.index.dense_ready else None,
            "embedding_model": self.index.embedder.name if self.index.dense_ready else None,
            "embedding_error": self.index.embedder.error if self.index.embedder else None,
            "llm": self.llm.provider,
            "llm_model": self.llm.model,
        }

    # ----------------------------------------------------------------- answer
    def _condense(self, question: str, history: list[dict]) -> str:
        """Ubah pertanyaan lanjutan menjadi pertanyaan mandiri (hanya jika ada LLM)."""
        if not history or not self.llm.enabled:
            return question
        recent = history[-4:]
        if len(tokenize(question)) >= 6:
            return question
        convo = "\n".join(f"{'Pengguna' if m.get('role') == 'user' else 'Asisten'}: {m.get('content', '')[:400]}" for m in recent)
        try:
            out = self.llm.generate(
                "Tulis ulang pertanyaan lanjutan menjadi satu pertanyaan mandiri berbahasa Indonesia yang lengkap "
                "berdasarkan percakapan. Jangan menjawab. Keluarkan hanya pertanyaannya.",
                f"Percakapan:\n{convo}\n\nPertanyaan lanjutan: {question}\n\nPertanyaan mandiri:",
                max_tokens=120,
            )
            out = out.strip().strip('"')
            return out if 3 <= len(out) <= 400 else question
        except LLMError:
            return question

    def _relevant(self, hits: list[Hit]) -> bool:
        if not hits:
            return False
        best = max(hits, key=lambda h: h.relevance)
        if best.dense is None:
            # BM25-only: false positive aman, LLM akan fallback ke mode general bila konteks tak memuat jawaban
            return best.coverage >= settings.MIN_COVERAGE
        strong_semantic = best.dense >= 0.55 and best.coverage >= 0.2
        return (best.coverage >= settings.MIN_COVERAGE and best.dense >= settings.MIN_DENSE_SIM) or strong_semantic

    @staticmethod
    def _context(hits: list[Hit]) -> str:
        blocks = []
        for i, h in enumerate(hits, 1):
            c = h.chunk
            pages = f"hal. {c.page_start}" if c.page_start == c.page_end else f"hal. {c.page_start}–{c.page_end}"
            sec = f", {c.section}" if c.section else ""
            blocks.append(f"[{i}] DOKUMEN: {c.doc_title} ({pages}{sec})\n{c.text}")
        return "\n\n---\n\n".join(blocks)

    @staticmethod
    def _snippet(text: str, q_terms: list[str], size: int = 340) -> str:
        """Potongan teks yang paling banyak memuat istilah pertanyaan."""
        lines = [l for l in text.splitlines() if l.strip()]
        qs = set(q_terms)
        best_i, best_s = 0, -1
        for i, l in enumerate(lines):
            s = len(qs & set(tokenize(l)))
            if s > best_s:
                best_i, best_s = i, s
        out = " ".join(lines[best_i : best_i + 4])
        return (out[:size].rstrip() + "…") if len(out) > size else out

    def _to_sources(self, hits: list[Hit], q_terms: list[str], only: list[int] | None = None) -> list[Source]:
        out: list[Source] = []
        for i, h in enumerate(hits, 1):
            if only is not None and i not in only:
                continue
            c = h.chunk
            out.append(
                Source(
                    n=i, doc_id=c.doc_id, doc_title=c.doc_title, doc_short=c.doc_short,
                    page_start=c.page_start, page_end=c.page_end, section=c.section,
                    snippet=self._snippet(c.text, q_terms), text=c.text, relevance=round(h.relevance, 3),
                    scores={
                        "keyword": round(h.keyword, 3),
                        "vector": None if h.dense is None else round(max(h.dense, 0.0), 3),
                        "rerank": round(h.rerank, 3),
                    },
                )
            )
        return out

    def _extractive(self, hits: list[Hit], q_terms: list[str]) -> tuple[str, list[int]]:
        """Jawaban tanpa LLM: rangkum baris paling relevan dari kutipan teratas, lengkap dengan rujukan."""
        qs = set(q_terms)
        parts, used = [], []
        for i, h in enumerate(hits[:3], 1):
            lines = [l for l in h.chunk.text.splitlines() if l.strip()]
            scored = [(len(qs & set(tokenize(l))), j) for j, l in enumerate(lines)]
            if not any(s for s, _ in scored):
                continue
            keep = sorted({j for s, j in scored if s > 0} | {j + 1 for s, j in scored if s > 1 and j + 1 < len(lines)})
            body = "\n".join(lines[j] for j in keep[:8])
            if len(body) > 900:
                body = body[:900].rstrip() + "…"
            c = h.chunk
            parts.append(f"**{c.doc_short}** (hal. {c.page_start}{', ' + c.section if c.section else ''}) [{i}]\n{body}")
            used.append(i)
        if not parts:
            return "", []
        head = "Berdasarkan dokumen yang ditemukan (mode ekstraktif, tanpa LLM):\n\n"
        return head + "\n\n".join(parts), used

    def ask(self, question: str, history: list[dict] | None = None, doc_ids: list[str] | None = None) -> Answer:
        t0 = time.perf_counter()
        history = history or []
        question = question.strip()
        meta: dict = {"llm": self.llm.provider, "retrieval": self.index.mode}
        doc_names = ""
        if self.docs:
            doc_names = " (" + "; ".join(d.short for d in self.docs.values()) + ")"

        def done(ans: Answer) -> Answer:
            ans.meta["latency_ms"] = int((time.perf_counter() - t0) * 1000)
            return ans

        if not self.ready:
            return done(Answer("Knowledge base belum siap. Silakan coba beberapa saat lagi.", False, meta=meta))

        # ── Kondense query berdasarkan riwayat ──
        prev_user = next((m.get("content", "") for m in reversed(history) if m.get("role") == "user"), "")
        standalone = self._condense(question, history)
        if standalone != question:
            meta["rewritten_query"] = standalone

        # ── Retrieve ──
        has_docs = bool(self.index.chunks)
        result = self.index.search(standalone, doc_ids=doc_ids, extra_context=prev_user if standalone == question else "") if has_docs else None
        hits = result.hits if result else []
        q_terms = result.query_terms if result else []
        if result:
            meta["routed_docs"] = result.routed_docs
            meta["route_reason"] = result.route_reason
            meta["top_relevance"] = round(max((h.relevance for h in hits), default=0.0), 3)

        is_relevant = has_docs and self._relevant(hits)

        # ── Mode GENERAL: pertanyaan umum, sapaan, atau tidak ada dokumen relevan ──
        if not is_relevant and self.llm.enabled:
            convo = ""
            if history:
                convo = "Riwayat percakapan sebelumnya:\n" + "\n".join(
                    f"{'Pengguna' if m.get('role') == 'user' else 'Asisten'}: {m.get('content', '')[:300]}"
                    for m in history[-6:]
                ) + "\n\n"
            user_prompt = f"{convo}Pertanyaan: {standalone}"
            # Sertakan info dokumen yang tersedia agar AI bisa mengarahkan
            if has_docs:
                doc_hint = "\n\nCatatan: knowledge base tersedia berisi: " + ", ".join(d.short for d in self.docs.values())
                user_prompt += doc_hint
            try:
                text = self._gen(SYSTEM_PROMPT_GENERAL, user_prompt)
                meta["mode"] = "general"
                meta["llm_model"] = getattr(self.llm, "_last_model", "")
                return done(Answer(text.strip(), True, [], meta))
            except LLMError as e:
                log.error("LLM general gagal: %s", e)
                meta["llm_error"] = str(e)[:200]
                # Coba jawab dari dokumen tanpa LLM bila ada potongan yang cukup mirip
                if hits and max(h.coverage for h in hits) >= 0.3:
                    text, used = self._extractive(hits, q_terms)
                    if text:
                        meta["llm"] = "extractive (fallback)"
                        return done(Answer(text, True, self._to_sources(hits, q_terms, used), meta))
                small = self._smalltalk(question)
                if small:
                    meta["mode"] = "offline"
                    return done(Answer(small, True, [], meta))
                meta["mode"] = "offline"
                return done(Answer(self._llm_down_message(str(e)), True, [], meta))

        # ── Mode NOT_FOUND: tidak relevan, LLM tidak tersedia ──
        if not is_relevant:
            meta["reason"] = "relevansi di bawah ambang"
            return done(Answer(NOT_FOUND_ANSWER.format(docs=doc_names), False, meta=meta))

        # ── Mode RAG: ada dokumen relevan ──
        if self.llm.enabled:
            convo = ""
            if history:
                convo = "RIWAYAT PERCAKAPAN (untuk memahami konteks pertanyaan saja):\n" + "\n".join(
                    f"{'Pengguna' if m.get('role') == 'user' else 'Asisten'}: {m.get('content', '')[:300]}" for m in history[-4:]
                ) + "\n\n"
            user_prompt = f"{convo}KONTEKS:\n{self._context(hits)}\n\nPERTANYAAN: {standalone}\n\nJAWABAN:"
            try:
                text = self._gen(SYSTEM_PROMPT_RAG, user_prompt)
                meta["mode"] = "rag"
                meta["llm_model"] = getattr(self.llm, "_last_model", "")
            except LLMError as e:
                log.error("LLM gagal: %s", e)
                meta["llm_error"] = str(e)[:200]
                text, used = self._extractive(hits, q_terms)
                meta["llm"] = "extractive (fallback)"
                if not text:
                    return done(Answer(NOT_FOUND_ANSWER.format(docs=doc_names), False, meta=meta))
                return done(Answer(text, True, self._to_sources(hits, q_terms, used), meta))

            if text.strip().upper().startswith(NOT_FOUND_TOKEN) or not text.strip():
                # RAG tidak menemukan → fallback ke general AI
                if self.llm.enabled:
                    try:
                        convo2 = ""
                        if history:
                            convo2 = "Riwayat:\n" + "\n".join(
                                f"{'Pengguna' if m.get('role') == 'user' else 'Asisten'}: {m.get('content', '')[:200]}"
                                for m in history[-4:]
                            ) + "\n\n"
                        gen_text = self._gen(SYSTEM_PROMPT_GENERAL, f"{convo2}Pertanyaan: {standalone}")
                        meta["mode"] = "general_fallback"
                        return done(Answer(gen_text.strip(), True, [], meta))
                    except LLMError:
                        pass
                meta["reason"] = "LLM: tidak ada dalam konteks"
                return done(Answer(NOT_FOUND_ANSWER.format(docs=doc_names), False, meta=meta))

            cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= len(hits)})
            return done(Answer(text.strip(), True, self._to_sources(hits, q_terms, cited or [1, 2, 3][: len(hits)]), meta))

        # ── Extractive fallback (tanpa LLM) ──
        text, used = self._extractive(hits, q_terms)
        if not text:
            meta["reason"] = "tidak ada kalimat relevan"
            return done(Answer(NOT_FOUND_ANSWER.format(docs=doc_names), False, meta=meta))
        return done(Answer(text, True, self._to_sources(hits, q_terms, used), meta))

