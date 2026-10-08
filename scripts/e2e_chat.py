import json, sys, time, urllib.request

BASE = "http://127.0.0.1:8080"


def ask(q, history=None):
    data = json.dumps({"message": q, "history": history or []}).encode()
    req = urllib.request.Request(BASE + "/api/chat", data=data, headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=180) as r:
        res = json.loads(r.read())
    return res, time.time() - t


# tunggu server siap
for _ in range(60):
    try:
        h = json.loads(urllib.request.urlopen(BASE + "/api/health", timeout=5).read())
        if h["status"] == "ok":
            print("HEALTH:", h["retrieval"], "| llm:", h["llm"], h["llm_model"], "| chunks:", h["chunks"])
            break
    except Exception:
        pass
    time.sleep(2)

tests = [
    "Apa persyaratan pangkat untuk mengikuti seleksi Sekolah Bintara Polisi dari Tamtama ke Bintara T.A. 2027 bagi lulusan SMA?",
    "Bagaimana ketentuan penilaian kinerja anggota Polri?",
    "Siapa presiden pertama Republik Indonesia?",
]
for q in tests:
    res, dt = ask(q)
    print("=" * 70)
    print("Q:", q)
    print("found=%s | llm=%s | %.1fs | docs=%s" % (res["found"], res["meta"].get("llm"), dt, res["meta"].get("routed_docs")))
    print(res["answer"][:700])
    print("sources:", [(s["n"], s["doc_short"][:25], s["page_start"]) for s in res["sources"]])
