import sys, time, threading
sys.path.insert(0, "/app")
from app.llm import LLM, LLMError, PROVIDERS

def mk():
    l = LLM()
    l._set_provider("gemini", "AIzaFAKEKEY000000000000", "m-slow")
    l._gem_ready.wait(3)          # discovery gagal (key palsu) -> daftar kosong
    l._gem_models = ["m-slow", "m-fast", "m-dead"]
    l._gem_ready = threading.Event(); l._gem_ready.set()
    return l

# 1. Balapan: model pertama macet 15 dtk, kedua 1 dtk -> total ~4 dtk (3 dtk jeda + 1)
l = mk()
def inv1(protocol, model, *a):
    if model == "m-slow": time.sleep(15); return "slow"
    if model == "m-fast": time.sleep(1); return "fast"
    raise LLMError("HTTP 404: gone")
l._invoke = inv1
t = time.time(); out = l.generate("s", "u"); dt = time.time() - t
print(f"1. balapan : out={out} dt={dt:.1f}s  (harus fast, ~4s)  {'OK' if out=='fast' and dt<6 else 'GAGAL'}")

# 2. Model tercepat dipelajari: panggilan berikut m-fast duluan -> ~1 dtk
t = time.time(); out = l.generate("s", "u"); dt = time.time() - t
print(f"2. belajar : out={out} dt={dt:.1f}s  (harus fast, ~1s)  {'OK' if out=='fast' and dt<2.5 else 'GAGAL'}")

# 3. Semua gagal -> breaker aktif; panggilan berikut GAGAL INSTAN (<0.1s)
l = mk()
def inv3(protocol, model, *a): raise LLMError("Gagal menghubungi Gemini: ReadTimeout")
l._invoke = inv3
try: l.generate("s", "u")
except LLMError as e: print("3a. gagal total :", str(e)[:60], "OK")
t = time.time()
try: l.generate("s", "u"); print("3b GAGAL: seharusnya error")
except LLMError as e:
    dt = time.time() - t
    print(f"3b. breaker : dt={dt:.3f}s usable={l.usable} pesan='{str(e)[:40]}...' {'OK' if dt<0.1 and not l.usable else 'GAGAL'}")

# 4. Pulih setelah breaker habis
l._down_until = 0.0
l._invoke = lambda protocol, model, *a: "pulih"
print("4. pulih   :", l.generate("s", "u"), "OK" if l.usable else "GAGAL")

# 5. Auth error (401) tidak memicu breaker dan langsung berhenti
l = mk()
def inv5(protocol, model, *a): raise LLMError("HTTP 401: key salah")
l._invoke = inv5
try: l.generate("s", "u")
except LLMError as e: print("5. auth     :", "HTTP 401" in str(e) and l.usable, "(harus True: pesan 401 utuh, breaker tidak aktif)")
