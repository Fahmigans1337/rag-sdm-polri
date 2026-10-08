import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import httpx
from app.config import settings

model = sys.argv[1] if len(sys.argv) > 1 else "gemini-3.5-flash"
level = sys.argv[2] if len(sys.argv) > 2 else "low"
ctx = "Pasal 2: Penilaian Kinerja anggota Polri dilakukan terhadap anggota mulai pangkat Bhayangkara Dua sampai Komisaris Jenderal Polisi. " * 15
body = {
    "systemInstruction": {"parts": [{"text": "Jawab hanya dari KONTEKS, sertakan rujukan [1]."}]},
    "contents": [{"role": "user", "parts": [{"text": f"KONTEKS:\n[1] {ctx}\n\nPERTANYAAN: Siapa saja yang dinilai kinerjanya?\n\nJAWABAN:"}]}],
    "generationConfig": {"temperature": 0.1, "maxOutputTokens": 900, "thinkingConfig": {"thinkingLevel": level}},
}
t = time.time()
try:
    r = httpx.post(f"{settings.GEMINI_BASE_URL}/models/{model}:generateContent", json=body,
                   headers={"x-goog-api-key": settings.GEMINI_API_KEY}, timeout=90)
    print("HTTP", r.status_code, "%.1fs" % (time.time() - t))
    j = r.json()
    if r.status_code == 200:
        c = j["candidates"][0]
        print("finishReason:", c.get("finishReason"))
        print("usage:", j.get("usageMetadata"))
        print("parts:", [(list(p.keys()), (p.get("text") or "")[:150]) for p in c.get("content", {}).get("parts", [])])
    else:
        print(json.dumps(j)[:300])
except Exception as e:
    print("ERR", type(e).__name__, "%.1fs" % (time.time() - t))
