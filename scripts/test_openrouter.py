"""Tes key OpenRouter: model mana yang menjawab."""
import sys, time, httpx

key = sys.argv[1]
base = "https://openrouter.ai/api/v1"
h = {"Authorization": f"Bearer {key}"}
try:
    r = httpx.get(f"{base}/key", headers=h, timeout=20)
    print("KEY INFO:", r.status_code, r.text[:300].replace("\n", " "))
except Exception as e:
    print("KEY INFO gagal:", e)
models = sys.argv[2:] or ["openai/gpt-4o-mini", "google/gemini-2.5-flash-lite", "meta-llama/llama-3.3-70b-instruct:free", "openai/gpt-oss-20b:free"]
for m in models:
    t = time.time()
    try:
        r = httpx.post(f"{base}/chat/completions", headers=h, timeout=40,
                       json={"model": m, "max_tokens": 30, "messages": [{"role": "user", "content": "Balas satu kata: halo"}]})
        if r.status_code == 200:
            print(f"OK   {m}  {time.time()-t:.1f}s ->", (r.json()["choices"][0]["message"]["content"] or "")[:40])
        else:
            print(f"FAIL {m}  HTTP {r.status_code}  {r.text[:140]}".replace("\n", " "))
    except Exception as e:
        print(f"FAIL {m}  {type(e).__name__}: {e}")
