"""Debug: cek skor BM25 untuk query RAG-PERPOL."""
import sys
sys.path.insert(0, '.')

from app.config import settings
from app.ingest import parse_file
from app.retrieval import KnowledgeIndex, tokenize

# Load docs
docs_dir = settings.DOCS_DIR
chunks = []
for f in sorted(docs_dir.glob("*.pdf")):
    try:
        d = parse_file(f)
        chunks.extend(d.chunks)
        print(f"Loaded: {f.name} → {len(d.chunks)} chunks")
    except Exception as e:
        print(f"Skip {f.name}: {e}")

idx = KnowledgeIndex()
idx.build(chunks)

queries = [
    "Jelaskan penilaian kinerja Polri menurut Perpol 1/2025",
    "penilaian kinerja anggota polri",
    "ketentuan penilaian kinerja polri",
    "SIPK sistem manajemen kinerja polri",
]

MIN = settings.MIN_COVERAGE + 0.12
print(f"\nThreshold: MIN_COVERAGE({settings.MIN_COVERAGE}) + 0.12 = {MIN:.2f}\n")

for q in queries:
    tokens = tokenize(q)
    print(f"Query: {q!r}")
    print(f"  Tokens: {tokens}")
    result = idx.search(q)
    if result.hits:
        top = result.hits[:3]
        for i, h in enumerate(top, 1):
            print(f"  Hit {i}: cov={h.coverage:.3f} bm25={h.bm25:.3f} rel={h.relevance:.3f} → {'PASS' if h.coverage >= MIN else 'FAIL'} | {h.chunk.section or '—'} | {h.chunk.text[:60]}")
    else:
        print("  No hits")
    print()
