"""Tes alur API key dari UI (server dijalankan TANPA key)."""
import os, subprocess, sys, time, json, httpx
from pathlib import Path

root = Path(__file__).resolve().parent.parent
env_file = root / ".env"
bak = root / ".env.bak_test"
keyfile = root / "data" / "index" / "llm_key.json"
if env_file.exists():
    env_file.rename(bak)
keyfile.unlink(missing_ok=True)
env = {k: v for k, v in os.environ.items() if not k.startswith(("OPENAI_", "GEMINI_"))}
env["LLM_PROVIDER"] = "openai"
proc = subprocess.Popen([sys.executable, "-X", "utf8", "-m", "uvicorn", "app.main:app", "--port", "8097"],
                        cwd=root, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
base = "http://127.0.0.1:8097"
ok = True
def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name, extra)
try:
    for _ in range(40):
        try:
            if httpx.get(base + "/api/health", timeout=2).json()["status"]["ready"]:
                break
        except Exception:
            pass
        time.sleep(1)
    st = httpx.get(base + "/api/health").json()["status"]
    check("health: llm_configured=False tanpa key", st["llm_configured"] is False)
    r = httpx.post(base + "/api/chat", json={"message": "halo"}, timeout=30).json()
    check("chat tanpa key tetap menjawab (offline)", r["meta"].get("mode") == "offline", r["answer"][:50])
    r = httpx.post(base + "/api/llm-key", json={"key": "abc"}, timeout=20)
    check("key terlalu pendek ditolak (422)", r.status_code == 422, r.status_code)
    r = httpx.post(base + "/api/llm-key", json={"key": "sk-or-v1-" + "0" * 64}, timeout=30)
    check("key palsu ditolak penyedia (400)", r.status_code == 400, r.json().get("detail"))
    check("key palsu TIDAK tersimpan", not keyfile.exists())
    st = httpx.get(base + "/api/health").json()["status"]
    check("health tetap llm_configured=False", st["llm_configured"] is False)
finally:
    proc.terminate()
    try: proc.wait(10)
    except Exception: proc.kill()

# penyimpanan & pemuatan ulang (validate=False agar tanpa jaringan)
sys.path.insert(0, str(root))
os.environ.pop("OPENAI_API_KEY", None)
from app import llm, config  # noqa: E402
msg = llm.apply_key("sk-or-v1-" + "a" * 40, validate=False)
check("apply_key mengenali OpenRouter", "OpenRouter" in msg and config.settings.OPENAI_BASE_URL.startswith("https://openrouter.ai"), msg)
check("key tersimpan di volume", keyfile.exists())
config.settings.OPENAI_API_KEY = ""
check("load_saved_key memulihkan key", llm.load_saved_key() and config.settings.OPENAI_API_KEY.startswith("sk-or-v1-aaaa"))
keyfile.unlink(missing_ok=True)
if bak.exists():
    bak.rename(env_file)
print("\nHASIL:", "SEMUA LULUS" if ok else "ADA YANG GAGAL")
sys.exit(0 if ok else 1)
