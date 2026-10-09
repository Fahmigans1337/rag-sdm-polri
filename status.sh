#!/bin/bash
echo "--- containers"
docker ps --format "{{.Names}} {{.Status}}"
echo "--- health"
curl -s localhost:8080/api/health | python3 -c "
import sys,json
d=json.load(sys.stdin)
s=d['status']
print('llm=',s.get('llm'),'configured=',s.get('llm_configured'),'ready=',s.get('ready'))
" 2>&1
echo "--- key file"
docker exec rag-sdm-polri python3 -c "
import json,pathlib
p=pathlib.Path('/app/data/index/llm_key.json')
if p.exists():
    d=json.loads(p.read_text())
    k=d.get('key','')
    print('provider=',d.get('provider'),'key_prefix=',k[:8],'len=',len(k))
else:
    print('NO KEY FILE')
" 2>&1
echo "--- logs"
docker logs --tail 40 rag-sdm-polri 2>&1 | grep -E "rag.llm|rag.engine|ERROR|WARNING" | cut -c1-280
