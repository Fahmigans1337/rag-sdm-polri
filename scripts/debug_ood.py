import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging, warnings
logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings('ignore')
from app.retrieval import KnowledgeIndex, tokenize
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

q = 'siapa presiden pertama indonesia'
tokens = tokenize(q)
print('Query tokens after stopwords/stemming:', tokens)
r = idx.search(q, top_k=3)
print('Hits:', len(r.hits), 'query_terms:', r.query_terms)
for h in r.hits[:3]:
    print(f'  cov={h.coverage:.3f} bm25={h.bm25:.3f} score={h.score:.4f}')
    print('  text:', h.chunk.text[:150])
