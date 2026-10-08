"""Jalankan OCR pada PDF scan dan simpan sebagai sidecar JSON (data/ocr/<nama>.json).

Pemakaian:  python scripts/ocr_pdf.py "data/docs/PENGUMUMAN SBP TA 2027.pdf"
"""
import json, logging, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings
from app.ingest import _ocr_pages

logging.basicConfig(level=logging.INFO, format="%(message)s")
src = Path(sys.argv[1])
pages = _ocr_pages(src)
settings.OCR_DIR.mkdir(parents=True, exist_ok=True)
out = settings.OCR_DIR / f"{src.stem}.json"
out.write_text(json.dumps([{"page": n, "text": t} for n, t in sorted(pages.items())], ensure_ascii=False, indent=1), encoding="utf-8")
print("Tersimpan:", out, f"({len(pages)} halaman)")
