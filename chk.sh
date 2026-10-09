#!/bin/bash
set -e
SRC=/mnt/c/Users/Unknown/.gemini/antigravity/scratch/rag-sdm-polri
cd ~/rag-sdm-polri
cp $SRC/app/llm.py app/llm.py
docker compose up -d --build 2>&1 | tail -2
docker exec rag-sdm-polri rm -f /app/data/index/llm_key.json
docker compose restart 2>&1 | tail -1
for i in $(seq 1 40); do curl -sf localhost:8080/api/health >/dev/null && break; sleep 2; done
echo "--- health setelah key tersimpan dihapus"
curl -s localhost:8080/api/health | python3 -c "import sys,json;s=json.load(sys.stdin)['status'];print(s['llm'],s['llm_configured'])"
echo "--- key palsu semua provider (harus DITOLAK semua)"
for k in "sk-or-v1-0000000000000000000000000000" "sk-000000000000000000000000000000" "gsk_00000000000000000000000000000" "sk-ant-00000000000000000000000000" "AIza0000000000000000000000000000000" "abcdefghijklmnopqrstuvwxyz123456" "tidak-dikenal-format-123456"; do
  echo -n "${k:0:12}... => "
  curl -s -X POST localhost:8080/api/llm-key -H 'Content-Type: application/json' -d "{\"key\":\"$k\"}" | head -c 160
  echo
done
echo "--- health akhir (harus belum configured)"
curl -s localhost:8080/api/health | python3 -c "import sys,json;s=json.load(sys.stdin)['status'];print(s['llm'],s['llm_configured'])"
echo "--- key file tersimpan?"
docker exec rag-sdm-polri ls /app/data/index/ | grep llm_key || echo "tidak ada (benar)"
