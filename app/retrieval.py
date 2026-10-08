"""Retrieval hibrida: BM25 (leksikal) + embedding multibahasa (semantik) + Reciprocal Rank Fusion.

Juga menyediakan *routing dokumen* agar konteks dari dokumen yang tidak relevan
tidak ikut terbawa ke jawaban.
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from .config import settings
from .ingest import Chunk

log = logging.getLogger("rag.retrieval")

# ------------------------------------------------------------------ text analysis
STOPWORDS = set(
    """
    yang dan di ke dari untuk pada dengan atau adalah itu ini oleh dalam sebagai juga akan
    dapat dapatkah apa apakah siapa bagaimana berapa kapan dimana mana mengapa kenapa saja
    ada tidak bukan sudah telah harus para serta bagi tentang terhadap antara agar bila jika
    apabila sehingga karena namun tetapi atas bahwa saya kami kita anda mereka dia nya lah kah
    tolong sebutkan apa-apa sebuah suatu setiap seluruh semua masing yaitu yakni
    dibawah diatas nomor bulan hari tanggal halaman ayat lampiran
    presiden wakil gubernur bupati walikota menteri jenderal sekretaris direktur kepala
    jelaskan ceritakan uraikan deskripsikan sebutkan bagikan sampaikan
    menurut mengenai terkait berdasarkan perihal seputar bagaimanakah apakah
    indonesia republik negara nasional umum resmi pertama kedua ketiga sesuai tahun pasal
    """.split()
)

_PREFIXES = ["meng", "meny", "mem", "men", "peng", "peny", "pem", "pen", "per", "ber", "ter", "me", "pe", "di", "ke", "se"]
_SUFFIXES = ["kan", "nya", "lah", "kah", "an", "i"]

# normalisasi singkatan/sinonim domain -> token kanonik (sebelum stemming)
SYNONYMS = {
    "persyaratan": "syarat",
    "prasyarat": "syarat",
    "kriteria": "syarat",
    "t.a.": "ta",
    "tahun anggaran": "ta",
    "tamtama": "tamtama",
    "bintara": "bintara",
    "sma": "sma",
    "smk": "sma",
    "sederajat": "sma",
    "personel": "anggota",
    "personil": "anggota",
    "pegawai": "anggota",
    "polisi": "polri",
    "kepolisian": "polri",
    "kinerja": "kinerja",
    "penilaian": "penilaian",
    "sipk": "sipk",
    "perpol": "perpol",
    "peraturan polri": "perpol",
    "peraturan kepolisian": "perpol",
    "smk": "sipk",
    "seleksi": "seleksi",
    "pendidikan": "pendidikan",
    "ketentuan": "ketentuan",
    "syarat": "syarat",
    "informasi": "informasi",
}

# Kata yang tidak boleh di-stem (domain-specific atau akan salah dipotong)
PROTECTED_WORDS = {
    "penilaian", "perpol", "kinerja", "polri", "sipk", "smk", "seleksi",
    "pendidikan", "ketentuan", "informasi", "anggota", "pangkat", "tamtama",
    "bintara", "bintara", "perwira", "pembinaan", "pengembangan", "pelaksanaan",
    "penyelenggaraan", "penerimaan", "pengumuman", "peraturan", "kepolisian",
    "penilai", "pejabat", "pelaksana", "pengguna", "pemegang",
}


def _stem(tok: str) -> str:
    if tok in PROTECTED_WORDS:
        return tok
    if len(tok) <= 4 or tok.isdigit():
        return tok
    for p in _PREFIXES:
        if tok.startswith(p) and len(tok) - len(p) >= 4:
            stripped = tok[len(p):]
            # Jangan potong jika hasilnya tidak bermakna (< 4 char)
            if len(stripped) >= 4:
                tok = stripped
                break
    for s in _SUFFIXES:
        if tok.endswith(s) and len(tok) - len(s) >= 4:
            tok = tok[: -len(s)]
            break
    return tok


_TOKEN = re.compile(r"[a-z0-9]+(?:[-/][a-z0-9]+)*", re.I)


def tokenize(text: str, keep_stop: bool = False) -> list[str]:
    text = text.lower().replace("t.a.", " ta ").replace("s-1", "s1").replace("s-2", "s2").replace("s-3", "s3")
    out: list[str] = []
    for raw in _TOKEN.findall(text):
        for tok in raw.replace("/", "-").split("-") if not raw[0].isdigit() else [raw]:
            if not tok:
                continue
            tok = SYNONYMS.get(tok, tok)
            if not keep_stop and tok in STOPWORDS:
                continue
            out.append(_stem(tok))
    return out


# -------------------------------------------------------------------------- BM25
class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.4, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = len(docs)
        self.tf = [Counter(d) for d in docs]
        self.dl = np.array([len(d) for d in docs], dtype=float)
        self.avgdl = float(self.dl.mean()) if self.N else 0.0
        df: Counter[str] = Counter()
        for c in self.tf:
            df.update(c.keys())
        self.df = df
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        self.max_idf = math.log(1 + (self.N + 0.5) / 0.5)

    def idf_of(self, term: str) -> float:
        # term yang tidak pernah muncul di korpus diberi idf maksimum (informatif sekaligus "asing")
        return self.idf.get(term, self.max_idf)

    def scores(self, query: list[str]) -> np.ndarray:
        s = np.zeros(self.N)
        for t in set(query):
            if t not in self.idf:
                continue
            idf = self.idf[t]
            for i, c in enumerate(self.tf):
                f = c.get(t)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.dl[i] / (self.avgdl or 1))
                s[i] += idf * f * (self.k1 + 1) / denom
        return s


# --------------------------------------------------------------------- embeddings
class Embedder:
    """Embedding lokal multibahasa via fastembed (ONNX, tanpa GPU/API key)."""

    def __init__(self) -> None:
        self.model = None
        self.name = settings.EMBEDDING_MODEL
        self.error: str | None = None
        if settings.EMBEDDING_BACKEND == "none":
            self.error = "dinonaktifkan (EMBEDDING_BACKEND=none)"
            return
        try:
            from fastembed import TextEmbedding

            self.model = TextEmbedding(model_name=self.name, cache_dir=settings.EMBEDDING_CACHE_DIR)
            log.info("Embedding model siap: %s", self.name)
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
            log.warning("Embedding tidak tersedia, memakai BM25 saja (%s)", self.error)
            if settings.EMBEDDING_BACKEND == "local":
                raise

    @property
    def ready(self) -> bool:
        return self.model is not None

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = np.array(list(self.model.embed(texts)), dtype=np.float32)  # type: ignore[union-attr]
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.clip(norms, 1e-9, None)


class VectorStore:
    """Penyimpanan vektor: Qdrant (mode lokal, persisten) dengan cadangan NumPy."""

    COLLECTION = "knowledge_base"

    def __init__(self) -> None:
        self.client = None
        self.matrix: np.ndarray | None = None
        self.doc_ids: list[str] = []
        self.backend = "numpy"

    def build(self, vectors: np.ndarray, doc_ids: list[str]) -> None:
        self.matrix, self.doc_ids = vectors, doc_ids
        try:
            from qdrant_client import QdrantClient, models

            settings.INDEX_DIR.mkdir(parents=True, exist_ok=True)
            if self.client is None:
                self.client = QdrantClient(path=str(settings.INDEX_DIR / "qdrant"))
            if self.client.collection_exists(self.COLLECTION):
                self.client.delete_collection(self.COLLECTION)
            self.client.create_collection(
                self.COLLECTION,
                vectors_config=models.VectorParams(size=int(vectors.shape[1]), distance=models.Distance.COSINE),
            )
            self.client.upsert(
                self.COLLECTION,
                points=[
                    models.PointStruct(id=i, vector=v.tolist(), payload={"doc_id": d})
                    for i, (v, d) in enumerate(zip(vectors, doc_ids))
                ],
            )
            self.backend = "qdrant"
        except Exception as e:  # noqa: BLE001
            self.client = None
            self.backend = "numpy"
            log.warning("Qdrant tidak dipakai (%s); memakai NumPy.", e)

    def search(self, qvec: np.ndarray, top: int) -> list[tuple[int, float]]:
        if self.matrix is None or not len(self.matrix):
            return []
        if self.client is not None:
            try:
                res = self.client.query_points(self.COLLECTION, query=qvec.tolist(), limit=top).points
                return [(int(p.id), float(p.score)) for p in res]
            except Exception as e:  # noqa: BLE001
                log.warning("Qdrant query gagal (%s); fallback NumPy", e)
        sims = self.matrix @ qvec
        idx = np.argsort(-sims)[:top]
        return [(int(i), float(sims[i])) for i in idx]


# --------------------------------------------------------------------- the index
@dataclass
class Hit:
    chunk: Chunk
    score: float  # skor fusi (RRF)
    coverage: float  # 0..1, cakupan istilah pertanyaan pada chunk (berbobot idf)
    dense: float | None  # cosine similarity (jika embedding aktif)
    bm25: float = 0.0
    keyword: float = 0.0  # skor leksikal gabungan (coverage + frasa + judul bagian)
    rerank: float = 0.0  # skor akhir hasil re-ranking (dipakai untuk mengurutkan)

    @property
    def relevance(self) -> float:
        """Skor gabungan 0..1 untuk membandingkan relevansi antar chunk/dokumen."""
        if self.dense is None:
            return self.coverage
        return 0.5 * self.coverage + 0.5 * max(self.dense, 0.0)


@dataclass
class RetrievalResult:
    hits: list[Hit]
    query_terms: list[str]
    routed_docs: list[str] = field(default_factory=list)
    route_reason: str = ""


class KnowledgeIndex:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.bm25: BM25 | None = None
        self.embedder: Embedder | None = None
        self.store = VectorStore()
        self.aliases: dict[str, list[str]] = {}  # doc_id -> alias (huruf kecil)

    # ---------- build
    def build(self, chunks: list[Chunk], aliases: dict[str, list[str]] | None = None) -> None:
        self.chunks = chunks
        self.aliases = {k: [a.lower() for a in v] for k, v in (aliases or {}).items()}
        self.bm25 = BM25([tokenize(c.embed_text) for c in chunks]) if chunks else None
        if self.embedder is None:
            self.embedder = Embedder()
        if chunks and self.embedder.ready:
            vecs = self._embed_cached([c.embed_text for c in chunks])
            self.store.build(vecs, [c.doc_id for c in chunks])
        log.info(
            "Index siap: %d chunk | BM25 | embedding=%s | vector store=%s",
            len(chunks), self.embedder.name if self.embedder.ready else "nonaktif", self.store.backend,
        )

    def _embed_cached(self, texts: list[str]) -> np.ndarray:
        """Cache embedding ke disk (kunci = model + hash isi) agar start berikutnya instan."""
        settings.INDEX_DIR.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256((self.embedder.name + "\n" + "\n".join(texts)).encode()).hexdigest()[:20]  # type: ignore[union-attr]
        f = settings.INDEX_DIR / f"emb_{key}.npy"
        if f.exists():
            try:
                return np.load(f)
            except Exception:  # noqa: BLE001
                pass
        vecs = self.embedder.encode(texts)  # type: ignore[union-attr]
        for old in settings.INDEX_DIR.glob("emb_*.npy"):
            old.unlink(missing_ok=True)
        np.save(f, vecs)
        return vecs

    @property
    def mode(self) -> str:
        return "hybrid (BM25 + embedding)" if self.embedder and self.embedder.ready else "BM25"

    # ---------- coverage
    def _coverage(self, q_terms: list[str], chunk_idx: int) -> float:
        assert self.bm25 is not None
        uniq = set(q_terms)
        if not uniq:
            return 0.0
        unseen = [t for t in uniq if t not in self.bm25.idf]
        # Singkatan rujukan (mis. "perpol", "1/2025") dibobot ringan, tetapi hanya bila mayoritas
        # token kueri dikenal korpus; kueri didominasi token asing (di luar dokumen) tetap ketat.
        unseen_w = 0.2 * self.bm25.max_idf if len(unseen) / len(uniq) < 0.5 else self.bm25.max_idf

        def w(t: str) -> float:
            return self.bm25.idf[t] if t in self.bm25.idf else unseen_w

        total = sum(w(t) for t in uniq)
        if total <= 0:
            return 0.0
        have = self.bm25.tf[chunk_idx]
        got = sum(w(t) for t in uniq if t in have)
        return got / total

    # ---------- routing
    def _alias_route(self, question: str) -> list[str]:
        q = question.lower()
        return [d for d, al in self.aliases.items() if any(a in q for a in al)]

    # ---------- search
    def search(
        self,
        query: str,
        top_k: int | None = None,
        doc_ids: list[str] | None = None,
        extra_context: str = "",
    ) -> RetrievalResult:
        top_k = top_k or settings.TOP_K
        if not self.chunks or self.bm25 is None:
            return RetrievalResult([], [])

        q_terms = tokenize(query)
        # pertanyaan lanjutan ("kalau untuk S-2?") dilengkapi konteks percakapan sebelumnya
        ext_terms = q_terms
        if extra_context and len(q_terms) < 4:
            ext_terms = q_terms + tokenize(extra_context)
        if not ext_terms:
            return RetrievalResult([], q_terms)

        n = len(self.chunks)
        bm = self.bm25.scores(ext_terms)
        dense = np.full(n, np.nan)
        dense_rank: dict[int, int] = {}
        if self.embedder and self.embedder.ready:
            qtext = query if not extra_context or len(q_terms) >= 4 else f"{extra_context} {query}"
            qvec = self.embedder.encode([qtext])[0]
            for r, (i, sim) in enumerate(self.store.search(qvec, min(40, n))):
                dense[i] = sim
                dense_rank[i] = r
            # similarity penuh untuk semua chunk (untuk routing & ambang)
            if self.store.matrix is not None:
                dense = self.store.matrix @ qvec

        bm_order = [int(i) for i in np.argsort(-bm) if bm[i] > 0][:40]
        bm_rank = {i: r for r, i in enumerate(bm_order)}

        K = 60
        fused: dict[int, float] = defaultdict(float)
        for i, r in bm_rank.items():
            fused[i] += 1.0 / (K + r)
        for i, r in dense_rank.items():
            fused[i] += 1.0 / (K + r)

        use_dense = bool(self.embedder and self.embedder.ready)
        hits_all: list[Hit] = []
        for i, sc in fused.items():
            hits_all.append(
                Hit(
                    chunk=self.chunks[i],
                    score=sc,
                    coverage=self._coverage(ext_terms, i),
                    dense=float(dense[i]) if use_dense else None,
                    bm25=float(bm[i]),
                )
            )
        hits_all.sort(key=lambda h: h.score, reverse=True)

        # ---- routing dokumen
        routed: list[str]
        reason: str
        if doc_ids:
            routed, reason = list(doc_ids), "dipilih pengguna"
        else:
            alias_docs = self._alias_route(query)
            if alias_docs:
                routed, reason = alias_docs, "disebut eksplisit pada pertanyaan"
            else:
                best: dict[str, float] = {}
                for h in hits_all[:20]:
                    best[h.chunk.doc_id] = max(best.get(h.chunk.doc_id, 0.0), h.relevance)
                if not best:
                    return RetrievalResult([], ext_terms)
                top_rel = max(best.values())
                routed = [d for d, v in best.items() if v >= 0.80 * top_rel]
                reason = "relevansi tertinggi"
        candidates = [h for h in hits_all if h.chunk.doc_id in routed][: max(top_k * 4, 24)]
        self._rerank(candidates, ext_terms)
        picked = sorted(candidates, key=lambda h: (h.rerank, h.score), reverse=True)[:top_k]
        return RetrievalResult(picked, ext_terms, routed, reason)

    # ---------- re-ranking (terinspirasi RAGFlow: multiple recall + fused re-ranking)
    @staticmethod
    def _rerank(hits: list[Hit], q_terms: list[str]) -> None:
        """Skor akhir = 0.3 * keyword-similarity + 0.7 * vector-similarity (jika ada vektor),
        keyword-similarity = 0.6 coverage istilah (idf) + 0.25 kecocokan frasa + 0.15 kecocokan judul bagian."""
        qset = set(q_terms)
        bigrams = {(a, b) for a, b in zip(q_terms, q_terms[1:])}
        for h in hits:
            toks = tokenize(h.chunk.text)
            phrase = 0.0
            if bigrams:
                found = {(a, b) for a, b in zip(toks, toks[1:]) if (a, b) in bigrams}
                phrase = len(found) / len(bigrams)
            sec_terms = set(tokenize(h.chunk.section or ""))
            sec = len(qset & sec_terms) / max(1, len(qset))
            h.keyword = 0.6 * h.coverage + 0.25 * phrase + 0.15 * sec
            h.rerank = h.keyword if h.dense is None else 0.3 * h.keyword + 0.7 * max(h.dense, 0.0)
