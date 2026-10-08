"""Konfigurasi aplikasi (semua lewat environment variable, default zero-config)."""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Muat KEY=VALUE dari file .env — selalu override agar key baru langsung terbaca."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if v:
            os.environ[k] = v  # override — .env selalu menang atas env lama


_load_dotenv(BASE_DIR / ".env")



def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    # --- Lokasi data -------------------------------------------------------
    DOCS_DIR: Path = Path(os.getenv("DOCS_DIR", BASE_DIR / "data" / "docs"))
    OCR_DIR: Path = Path(os.getenv("OCR_DIR", BASE_DIR / "data" / "ocr"))
    INDEX_DIR: Path = Path(os.getenv("INDEX_DIR", BASE_DIR / "data" / "index"))
    STATIC_DIR: Path = BASE_DIR / "app" / "static"

    # --- Chunking ----------------------------------------------------------
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1100"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "180"))

    # --- Retrieval ---------------------------------------------------------
    TOP_K: int = int(os.getenv("TOP_K", "6"))
    # auto  = pakai embedding lokal bila tersedia, jika gagal -> BM25 saja
    # local = wajib embedding lokal (fastembed), none = BM25 saja
    EMBEDDING_BACKEND: str = os.getenv("EMBEDDING_BACKEND", "auto").lower()
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    EMBEDDING_CACHE_DIR: str | None = os.getenv("EMBEDDING_CACHE_DIR") or None
    # Ambang relevansi: di bawah ini sistem menjawab "tidak ditemukan"
    MIN_COVERAGE: float = float(os.getenv("MIN_COVERAGE", "0.42"))
    MIN_DENSE_SIM: float = float(os.getenv("MIN_DENSE_SIM", "0.30"))

    # --- LLM ---------------------------------------------------------------
    # auto -> gemini (jika GEMINI_API_KEY) -> openai-compatible (jika OPENAI_API_KEY
    # atau OPENAI_BASE_URL) -> extractive (tanpa LLM, jawaban dirangkum dari kutipan)
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "auto").lower()
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    # dicoba berurutan jika model utama kena limit kuota / sibuk / tidak tersedia
    GEMINI_FALLBACK_MODELS: str = os.getenv(
        "GEMINI_FALLBACK_MODELS", "gemini-3.5-flash-lite,gemini-3.6-flash,gemini-3.8-flash,gemini-3.7-flash,gemini-3.1-flash-lite,gemini-flash-latest"
    )
    GEMINI_BASE_URL: str = os.getenv(
        "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
    )
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    LLM_TIMEOUT: float = float(os.getenv("LLM_TIMEOUT", "25"))

    AUTO_INGEST: bool = _bool("AUTO_INGEST", True)


settings = Settings()

