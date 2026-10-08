"""Tes cepat API key Gemini dari .env terhadap beberapa model."""
import httpx, pathlib, time

env = {}
for line in pathlib.Path(".env").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
key = env["GEMINI_API_KEY"]
print("KEY:", key[:14] + "...")
models = [env.get("GEMINI_MODEL", "gemini-3.5-flash-lite")] + env.get("GEMINI_FALLBACK_MODELS", "").split(",")
for m in [x.strip() for x in models if x.strip()]:
    t = time.time()
    try:
        r = httpx.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent",
            json={"contents": [{"role": "user", "parts": [{"text": "Balas satu kata: halo"}]}],
                  "generationConfig": {"maxOutputTokens": 20}},
            headers={"x-goog-api-key": key}, timeout=30)
        if r.status_code == 200:
            print(f"OK   {m}  {time.time()-t:.1f}s")
        else:
            msg = r.json().get("error", {}).get("message", r.text)[:110]
            print(f"FAIL {m}  HTTP {r.status_code}  {msg}")
    except Exception as e:
        print(f"ERR  {m}  {type(e).__name__}")
