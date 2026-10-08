import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging, warnings
logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings('ignore')
from app.retrieval import KnowledgeIndex
from app.ingest import parse_file
from app.config import settings
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
idx = KnowledgeIndex()
chunks = []
for fn in ['PERPOL NO 1 TAHUN 2025.pdf','PENGUMUMAN SBP TA 2027.pdf']:
    doc = parse_file(BASE / 'data/docs' / fn)
    chunks.extend(doc.chunks)
idx.build(chunks)

print('MIN_COVERAGE:', settings.MIN_COVERAGE, '(BM25-only threshold:', settings.MIN_COVERAGE + 0.25, ')')

# Mock RAGEngine._relevant logic
def relevant(hits):
    if not hits:
        return False
    best = max(hits, key=lambda h: h.relevance)
    if best.dense is None:
        return best.coverage >= settings.MIN_COVERAGE + 0.25
    strong = best.dense >= 0.55 and best.coverage >= 0.2
    return (best.coverage >= settings.MIN_COVERAGE and best.dense >= settings.MIN_DENSE_SIM) or strong

tests = [
    ('persyaratan pangkat SBP tamtama bintara', True),
    ('ketentuan penilaian kinerja anggota Polri SIPK', True),
    ('siapa presiden pertama indonesia', False),
    ('apa itu RAG machine learning', False),
    ('tahapan seleksi SBP 2027', True),
    ('apa saja syarat pendidikan untuk seleksi SBP', True),
    ('berapa lama pendidikan SBP', True),
]
ok_count = 0
for q, expected in tests:
    r = idx.search(q, top_k=3)
    hits = r.hits
    top_rel = max((h.relevance for h in hits), default=0.0)
    passes = relevant(hits)
    status = "OK" if (passes == expected) else "FAIL"
    if passes == expected:
        ok_count += 1
    print('[%s] passes=%s expected=%s rel=%.3f q=%r' % (status, passes, expected, top_rel, q[:55]))

print('\nHasil: %d/%d PASS' % (ok_count, len(tests)))
