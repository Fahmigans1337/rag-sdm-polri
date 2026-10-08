import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import httpx
from app.config import settings

models = sys.argv[1:] or ["gemini-3.8-flash", "gemini-3.5-flash", "gemini-flash-latest", "gemini-3.1-flash-lite"]
for m in models:
    url = f"{settings.GEMINI_BASE_URL}/models/{m}:generateContent"
    body = {"contents": [{"role": "user", "parts": [{"text": "Sebutkan ibu kota Indonesia dalam satu kata."}]}],
            "generationConfig": {"maxOutputTokens": 200}}
    t = time.time()
    try:
        r = httpx.post(url, json=body, headers={"x-goog-api-key": settings.GEMINI_API_KEY}, timeout=25)
        txt = r.text[:200].replace("\n", " ")
        if r.status_code == 200:
            txt = r.json()["candidates"][0]["content"]["parts"][0].get("text", "")[:80]
        print(f"{m}: HTTP {r.status_code} {time.time()-t:.1f}s -> {txt}")
    except Exception as e:
        print(f"{m}: ERR {type(e).__name__} {time.time()-t:.1f}s")
