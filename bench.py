import json, time, httpx

d = json.load(open("/app/data/index/llm_key.json"))
key = d["key"]
H = {"x-goog-api-key": key}
B = "https://generativelanguage.googleapis.com/v1beta"

r = httpx.get(f"{B}/models?pageSize=200", headers=H, timeout=20)
names = [m["name"].removeprefix("models/") for m in r.json()["models"]
         if "generateContent" in m.get("supportedGenerationMethods", []) and "flash" in m["name"]]
print("flash models:", names)

def call(model, extra=None, prompt="Siapa presiden pertama Indonesia? Jawab satu kalimat."):
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 300}}
    if extra:
        body["generationConfig"].update(extra)
    t = time.time()
    try:
        r = httpx.post(f"{B}/models/{model}:generateContent", json=body, headers=H,
                       timeout=httpx.Timeout(40, connect=8))
        dt = time.time() - t
        ok = r.status_code == 200
        txt = ""
        if ok:
            txt = "".join(p.get("text", "") for p in r.json()["candidates"][0]["content"]["parts"])[:50]
        else:
            txt = r.text[:90].replace("\n", " ")
        return f"{dt:5.1f}s HTTP {r.status_code} {txt}"
    except Exception as e:
        return f"{time.time()-t:5.1f}s ERR {type(e).__name__}"

for m in sorted(names)[:12]:
    if any(x in m for x in ("image", "tts", "live", "audio", "embed")):
        continue
    print(f"{m:38s} default      ->", call(m))
for m in [n for n in names if n.startswith("gemini-3")][:3]:
    print(f"{m:38s} think=low    ->", call(m, {"thinkingConfig": {"thinkingLevel": "low"}}))
    print(f"{m:38s} think=minimal->", call(m, {"thinkingConfig": {"thinkingLevel": "minimal"}}))
