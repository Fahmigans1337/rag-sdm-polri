import re, sys

P = sys.argv[1]
src = open(P, encoding="utf-8").read()
L = src.split("\n")

# ---------- 1. _call_gemini (baris 322-348) ----------
assert L[321].lstrip().startswith("def _call_gemini"), L[321]
assert "Respons Gemini tidak terduga" in L[347], L[347]
NEW_GEMINI = '''    def _call_gemini(self, model: str, system: str, user: str, temperature: float, max_tokens: int) -> str:
        base = PROVIDERS["gemini"]["base_url"]
        url = f"{base}/models/{model}:generateContent"
        gen: dict = {"temperature": temperature, "maxOutputTokens": max_tokens}
        if model.startswith("gemini-3"):
            gen["thinkingConfig"] = {"thinkingLevel": "minimal"}  # jawab cepat; penalaran panjang tidak perlu untuk tanya-jawab dokumen
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": gen,
        }
        r = None
        for attempt in range(2):
            try:
                r = httpx.post(url, json=body, headers={"x-goog-api-key": self.api_key}, timeout=self._timeout())
            except httpx.HTTPError as e:
                self._cooldown[model] = time.time() + 30
                raise LLMError(f"Gagal menghubungi Gemini: {type(e).__name__}") from e
            if r.status_code == 400 and "thinkingConfig" in gen and "thinking" in r.text.lower():
                gen.pop("thinkingConfig")  # model ini tidak mendukung pengaturan thinking: ulangi tanpa itu
                continue
            if r.status_code == 503 and attempt == 0:
                time.sleep(1.0)
                continue
            break
        if r.status_code != 200:
            wait_s = {429: 300, 503: 30, 404: 3600}.get(r.status_code, 30)
            self._cooldown[model] = time.time() + wait_s
            raise LLMError(f"HTTP {r.status_code}: {r.text[:200]}".replace("\\n", " "))
        try:
            parts = r.json()["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
        except (KeyError, IndexError) as e:
            raise LLMError(f"Respons Gemini tidak terduga: {r.text[:300]}") from e'''.split("\n")
L[321:348] = NEW_GEMINI

# ---------- 2. discover + generate (baris 156-231) ----------
assert "_discover_gemini_models" in L[155], L[155]
assert 'raise LLMError("Semua model gagal' in L[230], L[230]
NEW_CORE = '''    # Kebijakan kecepatan: latensi model sangat acak (1-40 dtk pada model yang sama), maka beberapa model
    # dipanggil "berlomba" dengan jeda HEDGE_DELAY dan jawaban tercepat dipakai. Model yang sering cepat diutamakan.
    HEDGE_DELAY = 3.0
    MAX_RACERS = 3
    BREAKER_SECONDS = 30.0

    @property
    def usable(self) -> bool:
        """Terkonfigurasi DAN tidak sedang dijeda circuit-breaker."""
        return self.enabled and time.time() >= self._down_until

    @staticmethod
    def _timeout() -> httpx.Timeout:
        return httpx.Timeout(min(settings.LLM_TIMEOUT, 20.0), connect=6.0)

    def _start_discovery(self) -> None:
        ev = threading.Event()
        self._gem_ready = ev
        key = self.api_key

        def work() -> None:
            try:
                found = self._discover_gemini_models(key)
                if self.api_key == key:
                    self._gem_models = found
                    if found:
                        self._cooldown = {}
            finally:
                ev.set()

        threading.Thread(target=work, daemon=True, name="gemini-models").start()

    def _discover_gemini_models(self, key: str | None = None) -> list[str]:
        """Ambil daftar model Gemini yang benar-benar tersedia untuk key ini (nama model sering diganti Google)."""
        try:
            r = httpx.get(
                f"{PROVIDERS['gemini']['base_url']}/models?pageSize=200",
                headers={"x-goog-api-key": key or self.api_key}, timeout=15,
            )
            if r.status_code != 200:
                return []
            full, lite = [], []
            for m in r.json().get("models", []):
                name = m.get("name", "").removeprefix("models/")
                if "generateContent" not in m.get("supportedGenerationMethods", []):
                    continue
                if not name.startswith("gemini-") or "flash" not in name:
                    continue
                if re.search(r"image|tts|live|audio|embed|thinking|exp|8b|robotics|computer|custom|learnlm|omni|latest", name):
                    continue
                v = re.match(r"gemini-(\\d+(?:\\.\\d+)?)", name)
                ver = float(v.group(1)) if v else 0.0
                (lite if "lite" in name else full).append((-ver, "preview" in name, name))
            full.sort()
            lite.sort()
            # selang-seling model penuh & lite agar balapan mencakup keluarga model berbeda
            ordered: list[str] = []
            for i in range(max(len(full), len(lite))):
                if i < len(full):
                    ordered.append(full[i][2])
                if i < len(lite):
                    ordered.append(lite[i][2])
            found = ordered[:6]
            log.info("Model Gemini tersedia: %s", found)
            return found
        except (httpx.HTTPError, ValueError, KeyError) as e:
            log.warning("Gagal mengambil daftar model Gemini: %s", type(e).__name__)
            return []

    def _candidates(self, protocol: str, p: dict) -> list[str]:
        static = [self.model] + [m for m in p.get("fallback_models", []) if m != self.model]
        if protocol == "gemini":
            self._gem_ready.wait(timeout=12)
            found = list(self._gem_models or [])
            pool = found + [m for m in static if m not in found]
        else:
            pool = static
        now = time.time()
        pool = [m for m in pool if self._cooldown.get(m, 0) <= now]
        known = sorted((m for m in pool if m in self._lat), key=lambda m: self._lat[m])
        unknown = [m for m in pool if m not in self._lat]
        return (known + unknown)[: self.MAX_RACERS + 1]

    def _invoke(self, protocol: str, model: str, system: str, user: str, temperature: float, max_tokens: int) -> str:
        if protocol == "gemini":
            return self._call_gemini(model, system, user, temperature, max_tokens)
        if protocol == "anthropic":
            return self._call_anthropic(model, system, user, temperature, max_tokens)
        return self._call_openai_compat(model, system, user, temperature, max_tokens)

    def generate(self, system: str, user: str, temperature: float = 0.2, max_tokens: int = 1500) -> str:
        if not self.enabled:
            raise LLMError("LLM tidak dikonfigurasi. Tempel API key via tombol API KEY.")
        if time.time() < self._down_until:
            raise LLMError("Layanan AI sedang dijeda sementara setelah gagal; jawaban memakai pencarian dokumen. Coba lagi sebentar.")

        p = PROVIDERS.get(self.provider_name)
        if not p:
            raise LLMError(f"Provider tidak dikenal: {self.provider_name}")
        protocol = p["protocol"]

        models = self._candidates(protocol, p)
        if not models:
            if protocol == "gemini":
                self._gem_models = None
                self._start_discovery()
            raise LLMError("Semua model sedang dijeda sementara setelah gagal berulang; coba lagi sebentar.")

        def run(m: str):
            t = time.time()
            return m, self._invoke(protocol, m, system, user, temperature, max_tokens), time.time() - t

        ex = ThreadPoolExecutor(max_workers=len(models), thread_name_prefix="llm")
        pending: dict = {}
        state = {"next": 0}
        errors: list[str] = []
        auth_failed = False
        end = time.time() + self._timeout().read + self.HEDGE_DELAY * (len(models) - 1) + 3

        def launch() -> None:
            m = models[state["next"]]
            state["next"] += 1
            pending[ex.submit(run, m)] = m

        launch()
        try:
            while pending:
                remaining = end - time.time()
                if remaining <= 0:
                    break
                more = state["next"] < len(models)
                done, _ = wait(list(pending), timeout=min(self.HEDGE_DELAY, remaining) if more else remaining,
                               return_when=FIRST_COMPLETED)
                if not done:
                    if more:
                        launch()  # model sebelumnya lambat: ikutkan model berikutnya (hedge)
                    continue
                failed = False
                for f in done:
                    m = pending.pop(f)
                    try:
                        _, out, dt = f.result()
                    except Exception as e:  # noqa: BLE001 - semua kegagalan satu model ditangani sama
                        log.warning("Model %s gagal: %s", m, str(e)[:200])
                        errors.append(f"{m}: {e}")
                        self._cooldown[m] = max(self._cooldown.get(m, 0), time.time() + 20)
                        failed = True
                        if "HTTP 401" in str(e) or "HTTP 403" in str(e):
                            auth_failed = True
                        continue
                    prev = self._lat.get(m)
                    self._lat[m] = dt if prev is None else 0.6 * prev + 0.4 * dt
                    self.model = m
                    self._last_model = m
                    self._down_until = 0.0
                    log.info("LLM %s menjawab dalam %.1f dtk", m, dt)
                    return out
                if auth_failed:
                    break
                if failed and state["next"] < len(models):
                    launch()
        finally:
            ex.shutdown(wait=False, cancel_futures=True)

        if protocol == "gemini" and errors and all("HTTP 404" in e for e in errors):
            self._gem_models = None  # daftar model berubah: temukan ulang di latar belakang
            self._start_discovery()
        elif not auth_failed:
            self._down_until = time.time() + self.BREAKER_SECONDS
        if not errors:
            raise LLMError("Layanan AI tidak merespons dalam batas waktu; jawaban memakai pencarian dokumen. Coba lagi sebentar.")
        raise LLMError("Semua model gagal | " + " | ".join(errors))'''.split("\n")
L[155:231] = NEW_CORE

out = "\n".join(L)

# ---------- 3. impor, state, set_provider, timeout ----------
out = out.replace("import time\n", "import threading\nimport time\nfrom concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait\n", 1)

old_init = "        self._gem_models: Optional[list[str]] = None\n"
assert old_init in out
out = out.replace(old_init, old_init +
    "        self._lat: dict[str, float] = {}  # rata-rata latensi (dtk) per model yang pernah sukses\n"
    "        self._down_until: float = 0.0  # circuit breaker: sampai kapan AI dijeda setelah gagal total\n"
    "        self._gem_ready = threading.Event()\n"
    "        self._gem_ready.set()\n", 1)

old_set = "        self._cooldown = {}\n        self._gem_models = None\n"
assert old_set in out
out = out.replace(old_set,
    "        self._cooldown = {}\n        self._gem_models = None\n        self._lat = {}\n        self._down_until = 0.0\n"
    "        if pname == \"gemini\":\n            self._start_discovery()  # cari model aktif di latar belakang (tidak memblokir pertanyaan pertama)\n"
    "        else:\n            self._gem_ready = threading.Event()\n            self._gem_ready.set()\n", 1)

assert out.count("timeout=settings.LLM_TIMEOUT") == 2, out.count("timeout=settings.LLM_TIMEOUT")
out = out.replace("timeout=settings.LLM_TIMEOUT", "timeout=self._timeout()")

open(P, "w", encoding="utf-8", newline="\n").write(out)
print("PATCH_OK lines:", out.count("\n") + 1)
