"""Test E2E lengkap: general chat + RAG dokumen."""
import httpx, time, sys

BASE = "http://127.0.0.1:8080"

# Tunggu index siap
print("Menunggu index siap...")
for i in range(30):
    try:
        r = httpx.get(f"{BASE}/api/health", timeout=10)
        st = r.json().get("status", {})
        ready = st.get("ready")
        chunks = st.get("chunks", 0)
        print(f"  [{i}] ready={ready} chunks={chunks}")
        if ready:
            break
    except Exception as e:
        print(f"  [{i}] error: {e}")
    time.sleep(3)
else:
    print("TIMEOUT: index tidak siap")
    sys.exit(1)

print()
TESTS = [
    ("Halo! Siapa kamu?", "SAPAAN"),
    ("Apa saja persyaratan mengikuti SBP T.A. 2027?", "RAG-SBP"),
    ("Siapa presiden Indonesia sekarang?", "UMUM"),
    ("Jelaskan penilaian kinerja Polri menurut Perpol 1/2025", "RAG-PERPOL"),
    ("Berapa jumlah planet di tata surya?", "UMUM-SAINS"),
]

for q, label in TESTS:
    r = httpx.post(f"{BASE}/api/chat", json={"message": q}, timeout=50)
    d = r.json()
    meta = d.get("meta", {})
    mode = meta.get("mode", "—")
    llm = meta.get("llm_model") or meta.get("llm", "—")
    found = d.get("found")
    src = len(d.get("sources", []))
    ms = meta.get("latency_ms", "?")
    ans = d.get("answer", "")[:130]
    print(f"[{label}]")
    print(f"  mode={mode} | llm={llm} | found={found} | src={src} | {ms}ms")
    print(f"  {ans}")
    print()
