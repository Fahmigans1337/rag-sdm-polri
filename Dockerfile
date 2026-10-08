FROM python:3.11-slim

# ---- System dependencies ----
# libgl1 & libglib2.0-0 dibutuhkan oleh PyMuPDF; curl untuk healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Python dependencies (layer di-cache) ----
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---- Source code & data ----
COPY app/ ./app/
COPY data/ ./data/

RUN mkdir -p data/docs data/index data/ocr

# ---- Non-root user ----
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# ---- Environment ----
# HF_HUB_*: unduhan model embedding gagal cepat bila offline (aplikasi tetap jalan dengan BM25)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PORT=8080 \
    HF_HUB_DOWNLOAD_TIMEOUT=20 \
    HF_HUB_ETAG_TIMEOUT=10 \
    HF_HUB_DISABLE_TELEMETRY=1

EXPOSE 8080

# ---- Health check ----
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=5 \
    CMD curl -f http://localhost:${PORT}/api/health || exit 1

# ---- Start server ----
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
