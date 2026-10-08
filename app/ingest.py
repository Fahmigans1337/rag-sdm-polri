"""Ekstraksi teks PDF/TXT dan chunking yang sadar struktur dokumen.

- PDF teks    : diekstrak langsung dengan PyMuPDF.
- PDF scan    : memakai OCR sidecar (data/ocr/<nama>.json) bila ada; bila tidak ada dan
                `rapidocr-onnxruntime` terpasang, OCR dijalankan otomatis.
- Peraturan   : dipecah per BAB/Pasal agar satu chunk = satu ketentuan utuh.
- Pengumuman  : dipecah per jendela teks (dengan overlap) dan tetap membawa nomor halaman.
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .config import settings

log = logging.getLogger("rag.ingest")

MIN_TEXT_CHARS = 40  # halaman dengan teks lebih sedikit dianggap scan


@dataclass
class Chunk:
    id: str
    doc_id: str
    doc_title: str
    doc_short: str
    page_start: int
    page_end: int
    section: str
    text: str

    @property
    def embed_text(self) -> str:
        """Teks yang diindeks: diberi header konteks agar chunk 'tahu' asalnya."""
        head = f"{self.doc_short}"
        if self.section:
            head += f" | {self.section}"
        return f"{head}\n{self.text}"

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class ParsedDoc:
    doc_id: str
    title: str
    short: str
    filename: str
    pages: list[tuple[int, str]]
    method: str = "text"  # text | ocr-sidecar | ocr | mixed
    chunks: list[Chunk] = field(default_factory=list)


# --------------------------------------------------------------------------- util
def slugify(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()
    return s or "dokumen"


def _load_catalog() -> dict:
    p = settings.DOCS_DIR / "catalog.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            log.warning("catalog.json tidak valid: %s", e)
    return {}


def _clean_line(line: str) -> str:
    line = line.replace("\u00a0", " ").replace("\u200b", "")
    line = re.sub(r"[ \t]+", " ", line).strip()
    return line


# --------------------------------------------------------------------- extraction
def _ocr_pages(pdf_path: Path, only: set[int] | None = None) -> dict[int, str]:
    """OCR halaman scan. Hanya jalan bila rapidocr-onnxruntime terpasang."""
    try:
        import numpy as np
        import pymupdf
        from rapidocr_onnxruntime import RapidOCR
    except Exception:  # noqa: BLE001
        log.warning("OCR tidak tersedia (pasang requirements-ocr.txt) untuk %s", pdf_path.name)
        return {}
    # ambang diturunkan agar baris bercetak tebal/miring (mis. teks tebal pada pengumuman) tidak hilang
    ocr = RapidOCR(text_score=0.3, box_thresh=0.3, unclip_ratio=1.8)
    out: dict[int, str] = {}
    with pymupdf.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            if only is not None and i not in only:
                continue
            pix = page.get_pixmap(dpi=250)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
            res, _ = ocr(img)
            out[i] = _ocr_to_text(res or [])
            log.info("OCR halaman %d selesai", i)
    return out


def _ocr_to_text(res: list) -> str:
    """Susun hasil OCR (kotak + teks) menjadi baris atas->bawah, kiri->kanan."""
    items = []
    for box, txt, _score in res:
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        items.append({"y": (min(ys) + max(ys)) / 2, "h": max(ys) - min(ys), "x": min(xs), "t": txt})
    items.sort(key=lambda d: d["y"])
    rows: list[list[dict]] = []
    for it in items:
        if rows:
            last = rows[-1]
            ref_y = sum(d["y"] for d in last) / len(last)
            ref_h = sum(d["h"] for d in last) / len(last)
            if abs(it["y"] - ref_y) <= 0.55 * max(ref_h, it["h"]):
                last.append(it)
                continue
        rows.append([it])
    return "\n".join(" ".join(d["t"] for d in sorted(r, key=lambda d: d["x"])) for r in rows)


def extract_pdf(path: Path) -> tuple[list[tuple[int, str]], str]:
    import pymupdf

    pages: list[tuple[int, str]] = []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            pages.append((i, page.get_text("text") or ""))

    scan_pages = {n for n, t in pages if len(t.strip()) < MIN_TEXT_CHARS}
    method = "text"
    if scan_pages:
        sidecar = settings.OCR_DIR / f"{path.stem}.json"
        ocr_map: dict[int, str] = {}
        if sidecar.exists():
            for item in json.loads(sidecar.read_text(encoding="utf-8")):
                ocr_map[int(item["page"])] = item["text"]
            method = "ocr-sidecar"
        missing = {n for n in scan_pages if not ocr_map.get(n, "").strip()}
        if missing:
            ocr_map.update(_ocr_pages(path, missing))
            method = "ocr" if method == "text" else method
        pages = [(n, ocr_map.get(n, t) if n in scan_pages else t) for n, t in pages]
        if len(scan_pages) < len(pages):
            method = f"mixed({method})"
    return pages, method


def extract_txt(path: Path) -> list[tuple[int, str]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    # satu "halaman" ~ 3000 karakter agar sitasi tetap punya penanda posisi
    parts = [text[i : i + 3000] for i in range(0, len(text), 3000)] or [""]
    return [(i + 1, p) for i, p in enumerate(parts)]


# ---------------------------------------------------------------------- cleaning
_PAGE_NUM = re.compile(r"^[-–—]?\s*\d{1,3}\s*[-–—]?$")


def _clean_pages(pages: list[tuple[int, str]]) -> list[tuple[int, list[str]]]:
    """Buang nomor halaman & header/footer yang berulang di banyak halaman."""
    cleaned: list[tuple[int, list[str]]] = []
    for n, t in pages:
        lines = [_clean_line(x) for x in t.splitlines()]
        lines = [x for x in lines if x and not _PAGE_NUM.match(x)]
        cleaned.append((n, lines))

    if len(cleaned) >= 4:
        counts: Counter[str] = Counter()
        for _, lines in cleaned:
            counts.update(set(l.lower() for l in lines[:4] + lines[-2:]))
        limit = max(3, int(len(cleaned) * 0.5))
        noisy = {l for l, c in counts.items() if c >= limit and len(l) < 90}
        cleaned = [(n, [l for l in lines if l.lower() not in noisy]) for n, lines in cleaned]
    return cleaned


# ---------------------------------------------------------------------- chunking
_BAB = re.compile(r"^BAB\s+([IVXLC]+)\b\s*$", re.I)
_PASAL = re.compile(r"^Pasal\s+(\d+[A-Z]?)\s*$", re.I)
_BAGIAN = re.compile(r"^Bagian\s+(Kesatu|Kedua|Ketiga|Keempat|Kelima|Keenam|Ketujuh|Kedelapan|Kesembilan|Kesepuluh)\b.*$", re.I)


def _window(lines: list[tuple[int, str]], size: int, overlap: int) -> list[list[tuple[int, str]]]:
    """Kelompokkan baris menjadi jendela ~size karakter dengan overlap berbasis baris."""
    out: list[list[tuple[int, str]]] = []
    cur: list[tuple[int, str]] = []
    cur_len = 0
    for ln in lines:
        cur.append(ln)
        cur_len += len(ln[1]) + 1
        if cur_len >= size:
            out.append(cur)
            keep: list[tuple[int, str]] = []
            kept = 0
            for prev in reversed(cur):
                if kept + len(prev[1]) > overlap:
                    break
                keep.insert(0, prev)
                kept += len(prev[1]) + 1
            cur, cur_len = keep, kept
    if cur and (not out or len(cur) > len(out[-1]) // 4 or sum(len(x[1]) for x in cur) > overlap + 40):
        out.append(cur)
    return out


_MARKER = re.compile(r"^(\(\d{1,2}\)|\d{1,2}[.)]|[a-zA-Z][.)]|[-•–])(\s|$)")


def _is_heading(l: str) -> bool:
    return bool(_BAB.match(l) or _PASAL.match(l) or _BAGIAN.match(l)) or (
        len(l) >= 8 and " " in l and l.isupper() and not l.endswith((",", ";", ":"))
    )


def _reflow(flat: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Gabungkan baris hasil pemenggalan PDF (justified text) menjadi paragraf/butir utuh.

    Baris baru dimulai hanya pada judul (BAB/Pasal/huruf kapital) atau penanda butir
    seperti `(1)`, `a.`, `2)`. Penanda yang berdiri sendiri digabung dengan baris sesudahnya.
    """
    out: list[list] = []  # [halaman, teks, keras?]
    for n, l in flat:
        hard = _is_heading(l)
        if not out or hard or out[-1][2] or _MARKER.match(l):
            out.append([n, l, hard])
        else:
            out[-1][1] += " " + l
    return [(n, t) for n, t, _ in out]


def build_chunks(doc: ParsedDoc) -> list[Chunk]:
    cleaned = _clean_pages(doc.pages)
    flat: list[tuple[int, str]] = [(n, l) for n, lines in cleaned for l in lines]
    if not flat:
        return []
    flat = _reflow(flat)

    legal = sum(1 for _, l in flat if _PASAL.match(l)) >= 3
    chunks: list[Chunk] = []

    def emit(group: list[tuple[int, str]], section: str) -> None:
        text = "\n".join(l for _, l in group).strip()
        if len(text) < 25:
            return
        chunks.append(
            Chunk(
                id=f"{doc.doc_id}#{len(chunks) + 1}",
                doc_id=doc.doc_id,
                doc_title=doc.title,
                doc_short=doc.short,
                page_start=group[0][0],
                page_end=group[-1][0],
                section=section,
                text=text,
            )
        )

    if legal:
        bab_label, bab_title_pending = "", False
        bagian = ""
        seg: list[tuple[int, str]] = []
        seg_label = "Pembukaan"

        def flush() -> None:
            nonlocal seg
            if not seg:
                return
            label = " › ".join(dict.fromkeys(x for x in (bab_label, bagian, seg_label) if x))
            if len("\n".join(l for _, l in seg)) <= settings.CHUNK_SIZE * 1.5:
                emit(seg, label)
            else:
                for i, g in enumerate(_window(seg, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP), 1):
                    emit(g, f"{label} (bagian {i})")
            seg = []

        bagian_title_pending = False
        for n, line in flat:
            if (m := _BAB.match(line)):
                flush()
                bab_label, bab_title_pending, bagian, bagian_title_pending = f"BAB {m.group(1).upper()}", True, "", False
                seg_label = bab_label
                continue
            if bab_title_pending and not _PASAL.match(line) and not _BAGIAN.match(line):
                # baris judul BAB (huruf kapital semua)
                if line.isupper() and len(line) < 120:
                    bab_label = f"{bab_label} {line.title()}"
                    continue
                bab_title_pending = False
            if (m := _BAGIAN.match(line)):
                flush()
                bagian = line.title()
                seg_label = ""
                bab_title_pending, bagian_title_pending = False, True
                continue
            if bagian_title_pending and not _PASAL.match(line):
                bagian_title_pending = False
                if len(line) < 100 and not line.endswith((".", ";", ":")):
                    bagian = f"{bagian} – {line}"
                    continue
            if (m := _PASAL.match(line)):
                flush()
                bab_title_pending = bagian_title_pending = False
                seg_label = f"Pasal {m.group(1)}"
                seg = [(n, line)]
                continue
            seg.append((n, line))
        flush()
    else:
        for i, g in enumerate(_window(flat, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP), 1):
            emit(g, _guess_section(g))
    return chunks


_NUM_ITEM = re.compile(r"^(\d{1,2})\.\s+(\S.{3,60})")


def _guess_section(group: list[tuple[int, str]]) -> str:
    """Untuk pengumuman: gunakan butir bernomor pertama sebagai label bagian."""
    for _, l in group[:6]:
        m = _NUM_ITEM.match(l)
        if m:
            return f"Butir {m.group(1)} – {m.group(2)[:50].rstrip()}"
    return ""


# ------------------------------------------------------------------- public API
def parse_file(path: Path, doc_id: str | None = None) -> ParsedDoc:
    catalog = _load_catalog()
    meta = catalog.get(path.name, {})
    doc_id = doc_id or meta.get("id") or slugify(path.stem)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        pages, method = extract_pdf(path)
    elif suffix in {".txt", ".md"}:
        pages, method = extract_txt(path), "text"
    else:
        raise ValueError(f"Format {suffix} belum didukung (gunakan PDF/TXT)")

    title = meta.get("title") or _guess_title(pages) or path.stem
    short = meta.get("short") or path.stem[:40]
    doc = ParsedDoc(doc_id=doc_id, title=title, short=short, filename=path.name, pages=pages, method=method)
    doc.chunks = build_chunks(doc)
    log.info("Dokumen %s: %d halaman, %d chunk (%s)", path.name, len(pages), len(doc.chunks), method)
    return doc


def _guess_title(pages: list[tuple[int, str]]) -> str:
    if not pages:
        return ""
    head = [l for l in (_clean_line(x) for x in pages[0][1].splitlines()) if l]
    return " ".join(head[:3])[:140]
