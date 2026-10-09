"""FastAPI: API chat RAG + manajemen dokumen + UI satu halaman."""
from __future__ import annotations

import logging
import re
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import settings
from .llm import LLMError, apply_key, load_saved_key, detect_provider, PROVIDERS
from .rag import RAGEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("rag.api")

engine = RAGEngine()
MAX_UPLOAD_MB = 60


@asynccontextmanager
async def lifespan(_: FastAPI):
    if load_saved_key(target=engine.llm):  # key yang ditempel lewat UI (tersimpan di volume)
        log.info("Key dari storage dimuat: provider=%s model=%s", engine.llm.provider_name, engine.llm.model)
    # Bangun index di thread terpisah agar server langsung merespons (/api/health menunjukkan status).
    if settings.AUTO_INGEST:
        threading.Thread(target=engine.load_all, daemon=True, name="ingest").start()
    else:
        engine.ready = True
    yield



app = FastAPI(
    title="RAG Merit SDM Polri",
    description="Tanya-jawab dokumen SDM Polri berbasis Retrieval-Augmented Generation.",
    version="1.0.0",
    lifespan=lifespan,
)


class Message(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=2, max_length=1000, description="Pertanyaan pengguna")
    history: list[Message] = Field(default_factory=list, max_length=20)
    document_ids: list[str] | None = Field(default=None)
    doc_ids: list[str] | None = Field(default=None)  # alias for new UI


@app.get("/api/health", tags=["Sistem"])
def health():
    st = engine.status()
    p = PROVIDERS.get(engine.llm.provider_name, {})
    return {
        "docs": engine.list_docs(),
        "status": {
            "ready": st["ready"],
            "error": st.get("error"),
            "docs": st["documents"],
            "chunks": st["chunks"],
            "retrieval": st.get("retrieval", "BM25"),
            "llm": engine.llm.provider_name,
            "llm_name": p.get("name", engine.llm.provider_name),
            "llm_model": engine.llm.model,
            "llm_error": engine.llm_error,
            "llm_configured": engine.llm.enabled,
        },
    }


class KeyRequest(BaseModel):
    key: str = Field(min_length=8, max_length=300)
    provider: str = Field(default="")  # opsional: override deteksi otomatis


@app.post("/api/llm-key", tags=["Sistem"])
def set_llm_key(req: KeyRequest):
    """Pasang API key LLM dari UI. Provider dikenali otomatis dari prefix key."""
    try:
        msg = apply_key(req.key, validate=True, provider_hint=req.provider or "", target=engine.llm)
    except LLMError as e:
        raise HTTPException(400, str(e)) from e
    engine.llm_error = None
    p = PROVIDERS.get(engine.llm.provider_name, {})
    return {
        "ok": True,
        "message": msg,
        "provider": engine.llm.provider_name,
        "provider_name": p.get("name", engine.llm.provider_name),
        "model": engine.llm.model,
    }


@app.post("/api/chat", tags=["Chat"])
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(422, "Pertanyaan tidak boleh kosong")
    doc_ids = req.document_ids or req.doc_ids or None
    ans = engine.ask(req.message, [m.model_dump() for m in req.history], doc_ids)
    return ans.to_dict()


@app.get("/api/documents", tags=["Dokumen"])
def documents():
    return {"documents": engine.list_docs(), "status": engine.status()}


@app.post("/api/documents", tags=["Dokumen"], status_code=201)
async def upload_document(file: UploadFile = File(...)):
    name = Path(file.filename or "").name
    if Path(name).suffix.lower() not in {".pdf", ".txt", ".md"}:
        raise HTTPException(415, "Hanya file PDF, TXT, atau MD yang didukung")
    safe = re.sub(r"[^\w .()-]", "_", name)
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Ukuran file maksimal {MAX_UPLOAD_MB} MB")
    if not data:
        raise HTTPException(400, "File kosong")
    settings.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    target = settings.DOCS_DIR / safe
    target.write_bytes(data)
    try:
        doc = engine.add_file(target)
    except ValueError as e:
        target.unlink(missing_ok=True)
        raise HTTPException(422, str(e)) from e
    except Exception as e:  # noqa: BLE001
        target.unlink(missing_ok=True)
        log.exception("Gagal ingest")
        raise HTTPException(500, f"Gagal memproses dokumen: {e}") from e
    return {"id": doc.doc_id, "title": doc.title, "pages": len(doc.pages), "chunks": len(doc.chunks), "method": doc.method}


@app.delete("/api/documents/{doc_id}", tags=["Dokumen"])
def delete_document(doc_id: str):
    docs = {d["id"]: d for d in engine.list_docs()}
    if doc_id not in docs:
        raise HTTPException(404, "Dokumen tidak ditemukan")
    if docs[doc_id]["builtin"]:
        raise HTTPException(403, "Dokumen bawaan knowledge base tidak dapat dihapus")
    engine.remove(doc_id)
    return {"deleted": doc_id}


@app.get("/api/documents/{doc_id}/chunks", tags=["Dokumen"])
def document_chunks(doc_id: str):
    chunks = engine.list_chunks(doc_id)
    if chunks is None:
        raise HTTPException(404, "Dokumen tidak ditemukan")
    return {"doc_id": doc_id, "count": len(chunks), "chunks": chunks}


@app.get("/api/documents/{doc_id}/file", tags=["Dokumen"])
def document_file(doc_id: str):
    d = engine.docs.get(doc_id)
    if not d:
        raise HTTPException(404, "Dokumen tidak ditemukan")
    path = settings.DOCS_DIR / d.filename
    if not path.exists():
        raise HTTPException(404, "File sumber tidak tersedia")
    return FileResponse(path, filename=d.filename, content_disposition_type="inline")


app.mount("/static", StaticFiles(directory=settings.STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(settings.STATIC_DIR / "index.html")
