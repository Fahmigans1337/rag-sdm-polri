#!/bin/bash
# Full health + crash check
echo "=== 1. CHAT ONLINE ==="
ask() {
  local t0=$(date +%s.%N)
  local r=$(curl -s -m 90 -X POST localhost:8080/api/chat \
    -H 'Content-Type: application/json' \
    -d "{\"message\":\"$1\"}" | python3 -c "
import sys,json
d=json.load(sys.stdin)
m=d.get('meta',{})
ans=d['answer'][:80].replace(chr(10),' ')
print(f\"mode={m.get('mode')} model={m.get('llm_model')} | {ans}\")
" 2>&1)
  local t1=$(date +%s.%N)
  printf "%5.1fs  [%s]\n         %s\n" $(echo "$t1 - $t0" | bc) "$1" "$r"
}
ask "halo"
ask "Siapa presiden pertama Republik Indonesia?"
ask "Apa saja persyaratan umum seleksi SBP Tamtama ke Bintara 2027?"
ask "Berapa batas usia pendaftar?"
ask "Apa itu SIPK pada penilaian kinerja Polri?"

echo ""
echo "=== 2. ENDPOINT LAINNYA ==="
echo -n "POST /api/llm-key (key kosong) -> "
curl -s -X POST localhost:8080/api/llm-key -H 'Content-Type: application/json' -d '{"key":""}' | head -c 120; echo
echo -n "POST /api/llm-key (key palsu pendek) -> "
curl -s -X POST localhost:8080/api/llm-key -H 'Content-Type: application/json' -d '{"key":"aaa"}' | head -c 120; echo
echo -n "GET  /api/docs -> "
curl -s localhost:8080/api/docs | python3 -c "import sys,json;d=json.load(sys.stdin);print(f'{len(d)} dokumen')" 2>&1
echo -n "GET  /        -> "
curl -s -o /dev/null -w "HTTP %{http_code}" localhost:8080/; echo

echo ""
echo "=== 3. CRASH CHECK (syntax + import) ==="
docker exec rag-sdm-polri python3 -c "
import ast, importlib, sys
sys.path.insert(0, '/app')
for f in ['app/llm.py','app/rag.py','app/main.py','app/config.py']:
    ast.parse(open(f'/app/{f}').read())
    print('SYNTAX_OK:', f)
for m in ['app.llm','app.rag','app.config']:
    importlib.import_module(m)
    print('IMPORT_OK:', m)
print('ALL_OK')
"

echo ""
echo "=== 4. THREADING LEAK CHECK ==="
docker exec rag-sdm-polri python3 -c "
import sys,time
sys.path.insert(0,'.')
import app.llm as L
llm=L.LLM()
import threading
before=threading.active_count()
for _ in range(5):
    try: llm.generate('s','u')
    except: pass
time.sleep(2)
after=threading.active_count()
print(f'Thread count before={before} after={after}  leak={max(0,after-before)} (harusnya 0-1)')
"

echo ""
echo "=== 5. LOG ERRORS TERAKHIR ==="
docker logs --tail 60 rag-sdm-polri 2>&1 | grep -E "ERROR|WARNING|CRITICAL" | grep -v "localhost" | cut -c1-280 | tail -20
