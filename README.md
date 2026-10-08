<div align="center">

<img src="app/static/img/logo.png" alt="Logo SDM Polri" width="120"/>

# RAG SDM POLRI

### Aplikasi Question & Answer berbasis **Retrieval-Augmented Generation (RAG)**
### untuk Dokumen SDM Kepolisian Negara Republik Indonesia

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![OpenAI](https://img.shields.io/badge/LLM-OpenAI_API-412991?style=flat-square&logo=openai&logoColor=white)](https://platform.openai.com)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

</div>

---

## 📋 Deskripsi

Prototype aplikasi **Question & Answer** yang memungkinkan personel Polri mendapatkan informasi dari dokumen SDM secara cepat dan akurat menggunakan bahasa sehari-hari — tanpa perlu membuka dokumen dan mencari halaman per halaman secara manual.

Sistem ini mengimplementasikan pipeline **RAG (Retrieval-Augmented Generation)** yang:
1. **Memproses** dokumen PDF menjadi potongan (*chunks*) terindeks
2. **Menemukan** bagian dokumen yang paling relevan dengan pertanyaan pengguna menggunakan BM25 + Hybrid Retrieval
3. **Menghasilkan** jawaban akurat berbahasa Indonesia menggunakan LLM (OpenAI), lengkap dengan referensi halaman sumber

> Sistem juga dapat menjawab pertanyaan umum di luar knowledge base sebagai asisten AI ramah.

---

## 🎯 Fitur Utama

| Fitur | Keterangan |
|---|---|
| 🔍 **Smart Retrieval** | BM25 lexical search + Hybrid Reranking ala RAGFlow |
| 🤖 **Dual Mode** | Mode RAG (dokumen) + Mode General (chat bebas) |
| 📄 **Multi-dokumen** | Upload PDF/TXT tak terbatas, dipilih otomatis sesuai pertanyaan |
| 📑 **Sitasi Halaman** | Setiap jawaban dilengkapi referensi `[1][2]` + nomor halaman |
| 🔎 **Chunk Explorer** | Inspect semua potongan dokumen yang terindeks |
| 📊 **Score Bars** | Visualisasi skor KEYWORD · VECTOR · RERANK per sumber |
| 🌐 **Cyber UI** | Tampilan terminal siber bertema SDM Polri (emas/merah/hitam) |
| 🐳 **Docker Ready** | Satu perintah `docker compose up` langsung jalan |
| 🔄 **LLM Fallback** | Rantai fallback otomatis jika model utama overloaded |
| 📱 **Responsive** | Tampilan desktop & mobile |

---

## 🏗️ Arsitektur Sistem

```
┌──────────────────────────────────────────────────────────────┐
│                        BROWSER / CLIENT                       │
│              Cyber UI (HTML + Vanilla JS + Canvas)            │
└───────────────────────────┬──────────────────────────────────┘
                            │ HTTP/REST
┌───────────────────────────▼──────────────────────────────────┐
│                   FastAPI Backend (:8080)                      │
│  POST /api/chat   GET /api/health   GET /api/documents        │
│  POST /api/documents   DELETE /api/documents/{id}             │
│  GET /api/documents/{id}/chunks   GET /api/documents/{id}/file│
└──────┬──────────────────────────────────────────┬────────────┘
       │                                          │
┌──────▼──────────────┐              ┌────────────▼────────────┐
│   RAG Engine        │              │   LLM (OpenAI API)       │
│                     │              │                          │
│  ┌───────────────┐  │              │  gpt-4o-mini            │
│  │  Ingest       │  │              │  ↓ fallback chain        │
│  │  PDF → Chunks │  │              │  (model lain via .env) │
│  │  Laws/General │  │              │                         │
│  └───────┬───────┘  │              │                         │
│          │          │              └──────────────────────────┘
│  ┌───────▼───────┐  │
│  │  Knowledge    │  │
│  │  Index        │  │
│  │  BM25 + RRF   │  │
│  │  Re-ranking   │  │
│  └───────┬───────┘  │
│          │          │
│  ┌───────▼───────┐  │
│  │  Answer Gen   │  │
│  │  RAG / General│  │
│  │  + Sitasi     │  │
│  └───────────────┘  │
└─────────────────────┘
```

---

## 🚀 Cara Menjalankan

### Prasyarat

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/Mac/Linux)
- API key LLM milik Anda sendiri — [OpenRouter](https://openrouter.ai/settings/keys) (disarankan), [OpenAI](https://platform.openai.com/api-keys), atau [Groq](https://console.groq.com/keys) (gratis). Lihat catatan di bawah

### 1. Clone Repository

```bash
git clone https://github.com/Fahmigans1337/rag-sdm-polri.git
cd rag-sdm-polri
```

### 2. Jalankan dengan Docker

```bash
docker compose up -d --build
```

Tunggu sekitar satu menit, lalu buka **http://localhost:8080**.

### 3. Tempel API key milik Anda (di browser)

Saat pertama dibuka, muncul jendela **"Aktifkan AI dengan API key Anda"** (atau klik tombol **🔑 API KEY**
di pojok kanan atas). Tempel key Anda, lalu klik **Aktifkan**. Key divalidasi ke penyedia, jenis penyedia
dikenali otomatis (OpenRouter `sk-or-v1-...`, OpenAI `sk-...`, Groq `gsk_...`), dan tersimpan di volume Docker
lokal sehingga tetap aktif setelah container di-restart. Tidak perlu terminal maupun file `.env`.

Alternatif lewat terminal (opsional):

```bash
OPENAI_API_KEY="tempel_key_anda" docker compose up -d --build            # Linux / macOS / WSL
```

```powershell
$env:OPENAI_API_KEY="tempel_key_anda"; docker compose up -d --build       # Windows PowerShell
```

> [!IMPORTANT]
> **Mengapa API key tidak ada di repository?** Penyedia API (OpenAI, Google, OpenRouter) memindai GitHub
> secara otomatis dan **langsung menonaktifkan key yang terlihat publik**. Karena itu key tidak disertakan
> di repo ini; setiap penguji memakai key miliknya sendiri. Tanpa key aplikasi tetap berjalan tanpa galat
> dalam **mode ekstraktif** (badge "AI OFFLINE", jawaban dokumen tetap muncul lengkap dengan sumber halaman).

<details>
<summary>Alternatif: skrip interaktif (key diketik tersembunyi, tidak masuk riwayat terminal)</summary>

```bash
chmod +x start.sh && ./start.sh        # Linux / macOS / WSL
```

```powershell
.\start.ps1                             # Windows PowerShell
```

Atau lewat file: `cp .env.example .env`, isi `OPENAI_API_KEY`, lalu `docker compose up -d --build`.

</details>

**Ganti key / hentikan:** jalankan ulang perintah di atas dengan key baru (container dibuat ulang otomatis),
atau `docker compose down` untuk menghentikan.

### 4. Pantau Log (opsional)

```bash
docker compose logs -f
```

Tunggu hingga log menampilkan:
```
INFO  rag.retrieval: Index siap: 131 chunk | BM25
INFO  uvicorn: Application startup complete.
```

### 4. Akses Aplikasi

Buka browser dan akses:

```
http://localhost:8080
```

---

## 🐳 Docker — Detail

### Struktur Container

```
Container: rag-sdm-polri
├── Port  : 8080 → 8080
├── Volume: rag_docs  → /app/data/docs   (dokumen PDF)
└── Volume: rag_index → /app/data/index  (BM25 & embedding cache)
```

### Perintah Berguna

```bash
# Jalankan di background
docker compose up -d

# Lihat log real-time
docker compose logs -f

# Hentikan
docker compose down

# Rebuild dari awal (setelah update kode)
docker compose up --build

# Hapus semua data (termasuk volume)
docker compose down -v
```

### Variabel Environment

| Variabel | Default | Keterangan |
|---|---|---|
| `OPENAI_API_KEY` | *(wajib)* | API Key dari platform.openai.com |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model OpenAI yang dipakai |
| `LLM_PROVIDER` | `openai` | Penyedia LLM (`openai` / `none` untuk mode ekstraktif) |
| `MIN_COVERAGE` | `0.42` | Ambang relevansi BM25 (0.0–1.0) |
| `TOP_K` | `6` | Jumlah chunk terambil per query |
| `LLM_TIMEOUT` | `25` | Timeout request ke LLM (detik) |
| `CHUNK_SIZE` | `1100` | Ukuran chunk teks (karakter) |

---

## ⚙️ Menjalankan Tanpa Docker (Development)

```bash
# Buat virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Install dependensi
pip install -r requirements.txt

# Jalankan server
python -m uvicorn app.main:app --host 127.0.0.1 --port 8080 --reload
```

---

## 📁 Struktur Proyek

```
rag-sdm-polri/
├── app/
│   ├── main.py          # FastAPI routes & lifespan
│   ├── rag.py           # RAG pipeline utama (ingest→retrieve→generate)
│   ├── retrieval.py     # BM25 + Hybrid Index + Re-ranking
│   ├── ingest.py        # PDF parser + chunking (Laws/General template)
│   ├── llm.py           # OpenAI client (+ fallback ekstraktif tanpa LLM)
│   ├── config.py        # Settings dari environment variable
│   └── static/
│       ├── index.html   # Single-page Cyber UI
│       ├── css/         # (embedded di index.html)
│       └── img/
│           └── logo.png # Logo SDM Polri
├── data/
│   ├── docs/            # Dokumen PDF knowledge base
│   ├── index/           # Cache BM25 & embedding (auto-generated)
│   └── catalog.json     # Metadata dokumen bawaan
├── scripts/
│   ├── test_retrieval.py    # Unit test retrieval (7 queries)
│   ├── test_e2e_full.py     # End-to-end test API
│   └── debug_retrieval.py   # Debug skor BM25 per query
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 🔧 Teknologi yang Digunakan

| Komponen | Teknologi |
|---|---|
| **Backend** | Python 3.11, FastAPI, Uvicorn |
| **PDF Parsing** | PyMuPDF (fitz) + OCR sidecar |
| **Retrieval** | BM25 (custom), Numpy, Reciprocal Rank Fusion |
| **Re-ranking** | Keyword coverage + phrase match + section match |
| **LLM** | OpenAI API (gpt-4o-mini) |
| **Frontend** | Vanilla JS, HTML5 Canvas (particle network) |
| **Containerisasi** | Docker, Docker Compose |
| **HTTP Client** | HTTPX (async-capable) |

---

## 💬 Cara Penggunaan

### Bertanya dari Knowledge Base

Ketik pertanyaan langsung di kolom teks, contoh:

```
Bagaimana ketentuan penilaian kinerja anggota Polri?
Apa saja persyaratan mengikuti SBP T.A. 2027?
Berapa pangkat minimal untuk seleksi Bintara dari Tamtama lulusan SMA?
```

Sistem akan:
1. Mencari bagian dokumen yang paling relevan
2. Memberikan jawaban dengan sitasi `[1][2]` dan nomor halaman
3. Menampilkan kartu sumber dengan skor relevansi

### Upload Dokumen Baru

Klik tombol **"Tambah dokumen"** di sidebar → pilih file PDF/TXT → tunggu proses ingest selesai.

### Pertanyaan Umum

Sistem juga dapat menjawab pertanyaan di luar knowledge base:

```
Halo, siapa kamu?
Apa itu RAG?
Berapa planet di tata surya?
```

### Chunk Explorer

Klik tombol **"Chunks"** pada dokumen di sidebar untuk melihat semua potongan teks yang terindeks.

---

## 🧪 Testing

```bash
# Test retrieval (7 query — harus semua PASS)
python scripts/test_retrieval.py

# Test end-to-end API (5 skenario)
python scripts/test_e2e_full.py

# Debug skor BM25 untuk query tertentu
python scripts/debug_retrieval.py
```

---

## 📝 Catatan

- Sistem menggunakan **BM25-only** (leksikal) karena koneksi ke model embedding eksternal mungkin timeout tergantung jaringan. Untuk mengaktifkan hybrid embedding, pastikan koneksi ke HuggingFace tersedia.
- Jawaban hanya berdasarkan dokumen di knowledge base. Jika informasi tidak tersedia, sistem akan menyampaikan bahwa data tidak ditemukan.
- API Key OpenAI dibuat di [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)

---

## 👨‍💻 Dibuat untuk

**Seleksi Kemampuan Pemrograman — Tahap 2**
Topik: *Retrieval-Augmented Generation (RAG)*
Periode: 8–14 Oktober 2026

---

<div align="center">

**RAG SDM POLRI** — *Secure Knowledge Terminal*

Built with ❤️ using FastAPI + OpenAI API

</div>
