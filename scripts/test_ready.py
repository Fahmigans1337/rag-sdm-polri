"""Tes: berapa detik sampai siap, lalu chat langsung (embedding boleh masih loading)."""
import sys, time, httpx

base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
t0 = time.time()
while True:
    try:
        st = httpx.get(f"{base}/api/health", timeout=5).json()["status"]
        if st["ready"]:
            break
    except Exception:
        pass
    if time.time() - t0 > 180:
        print("GAGAL: tidak siap dalam 180 detik"); sys.exit(1)
    time.sleep(1)
print(f"SIAP dalam {time.time()-t0:.1f}s | chunks={st['chunks']} | retrieval={st['retrieval']} | llm={st['llm_model']}")

for q in ["halo", "Apa saja persyaratan mengikuti SBP T.A. 2027?", "Bagaimana ketentuan penilaian kinerja anggota Polri?"]:
    t = time.time()
    r = httpx.post(f"{base}/api/chat", json={"message": q}, timeout=90)
    d = r.json()
    print(f"[{r.status_code}] {time.time()-t:.1f}s mode={d['meta'].get('mode')} llm={d['meta'].get('llm_model') or d['meta'].get('llm')} src={len(d['sources'])} :: {d['answer'][:90]!r}")
