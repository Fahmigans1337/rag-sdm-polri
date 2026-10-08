"""Tes cepat key OpenAI dari .env (OPENAI_API_KEY / OPENAI_MODEL)."""
import sys, time, httpx
from pathlib import Path

env = {}
for line in Path(__file__).resolve().parent.parent.joinpath(".env").read_text(encoding="utf-8").splitlines():
    if "=" in line and not line.lstrip().startswith("#"):
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
key = env.get("OPENAI_API_KEY", "")
base = (env.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
print("KEY:", key[:12] + "...", "| base:", base)
for model in [env.get("OPENAI_MODEL", "gpt-4o-mini"), "gpt-4o-mini", "gpt-4.1-mini"]:
    t = time.time()
    try:
        r = httpx.post(f"{base}/chat/completions", timeout=30,
                       headers={"Authorization": f"Bearer {key}"},
                       json={"model": model, "max_tokens": 20,
                             "messages": [{"role": "user", "content": "Balas satu kata: halo"}]})
        if r.status_code == 200:
            print(f"OK   {model}  {time.time()-t:.1f}s  ->", r.json()["choices"][0]["message"]["content"][:40])
        else:
            print(f"FAIL {model}  HTTP {r.status_code}  {r.text[:160]}")
    except Exception as e:
        print(f"FAIL {model}  {type(e).__name__}: {e}")
