"""Penyedia LLM: Gemini, OpenAI, OpenRouter, Groq, Mistral, Anthropic Claude, atau ekstraktif tanpa LLM."""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Optional

import httpx

from .config import settings

log = logging.getLogger("rag.llm")


class LLMError(RuntimeError):
    pass


@dataclass
class LLMInfo:
    provider: str
    model: str


# ─── Registry semua provider yang didukung ────────────────────────────────────
PROVIDERS = {
    "openrouter": {
        "name": "OpenRouter",
        "key_prefix": "sk-or-",
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "openai/gpt-4o-mini",
        "fallback_models": ["google/gemini-2.5-flash-lite", "meta-llama/llama-3.3-8b-instruct:free"],
        "protocol": "openai",
    },
    "openai": {
        "name": "OpenAI",
        "key_prefix": "sk-",
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "fallback_models": [],
        "protocol": "openai",
    },
    "groq": {
        "name": "Groq",
        "key_prefix": "gsk_",
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
        "fallback_models": ["llama-3.1-8b-instant", "gemma2-9b-it"],
        "protocol": "openai",
    },
    "mistral": {
        "name": "Mistral AI",
        "key_prefix": "mistral-",  # custom prefix kita pakai untuk deteksi manual
        "base_url": "https://api.mistral.ai/v1",
        "default_model": "mistral-small-latest",
        "fallback_models": ["open-mistral-7b"],
        "protocol": "openai",
    },
    "anthropic": {
        "name": "Anthropic (Claude)",
        "key_prefix": "sk-ant-",
        "base_url": "https://api.anthropic.com",
        "default_model": "claude-3-haiku-20240307",
        "fallback_models": [],
        "protocol": "anthropic",
    },
    "gemini": {
        "name": "Google Gemini",
        "key_prefix": "AIza",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "default_model": "gemini-2.5-flash",
        "fallback_models": ["gemini-2.5-flash-lite", "gemini-2.0-flash"],
        "protocol": "gemini",
    },
}


def detect_provider(key: str) -> Optional[str]:
    """Deteksi provider dari prefix API key secara otomatis."""
    key = key.strip()
    if key.startswith("sk-or-"):
        return "openrouter"
    if key.startswith("sk-ant-"):
        return "anthropic"
    if key.startswith("AIza") or key.startswith("AQ."):
        return "gemini"
    if key.startswith("gsk_"):
        return "groq"
    if key.startswith("sk-"):
        return "openai"
    # Mistral: panjang 32 hex (bukan prefix unik, deteksi via panjang + hex chars)
    if re.match(r'^[0-9a-zA-Z]{32}$', key):
        return "mistral"
    return None


class LLM:
    """Antarmuka tunggal multi-provider: generate(system, user) -> str."""

    def __init__(self) -> None:
        self.provider_name: str = "extractive"  # nama provider aktif
        self.model: str = ""
        self.api_key: str = ""
        self.base_url: str = ""
        self._cooldown: dict[str, float] = {}
        self._gem_models: Optional[list[str]] = None
        self._lat: dict[str, float] = {}  # rata-rata latensi (dtk) per model yang pernah sukses
        self._down_until: float = 0.0  # circuit breaker: sampai kapan AI dijeda setelah gagal total
        self._gem_ready = threading.Event()
        self._gem_ready.set()

        # Inisialisasi dari environment variable
        self._init_from_env()

    def _init_from_env(self) -> None:
        """Coba setiap provider dari env var."""
        # Prioritas: OPENAI_API_KEY -> GEMINI_API_KEY -> ekstraktif
        if settings.OPENAI_API_KEY:
            key = settings.OPENAI_API_KEY
            pname = detect_provider(key)
            if pname:
                self._set_provider(pname, key)
                return
        if settings.GEMINI_API_KEY:
            self._set_provider("gemini", settings.GEMINI_API_KEY)
            return

    def _set_provider(self, pname: str, key: str, model: str = "") -> None:
        """Set provider aktif."""
        if pname not in PROVIDERS:
            raise LLMError(f"Provider tidak dikenal: {pname}")
        p = PROVIDERS[pname]
        self.provider_name = pname
        self.api_key = key
        self.base_url = p["base_url"]
        self.model = model or p["default_model"]
        self._last_model = self.model
        self._cooldown = {}
        self._gem_models = None
        self._lat = {}
        self._down_until = 0.0
        if pname == "gemini":
            self._start_discovery()  # cari model aktif di latar belakang (tidak memblokir pertanyaan pertama)
        else:
            self._gem_ready = threading.Event()
            self._gem_ready.set()

    @property
    def provider(self) -> str:
        return self.provider_name

    @provider.setter
    def provider(self, val: str) -> None:
        self.provider_name = val

    @property
    def enabled(self) -> bool:
        return self.provider_name != "extractive" and bool(self.api_key)

    @property
    def info(self) -> LLMInfo:
        return LLMInfo(self.provider_name, self.model)

    # Kebijakan kecepatan: latensi model sangat acak (1-40 dtk pada model yang sama), maka beberapa model
    # dipanggil "berlomba" dengan jeda HEDGE_DELAY dan jawaban tercepat dipakai. Model yang sering cepat diutamakan.
    HEDGE_DELAY = 2.0
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
                v = re.match(r"gemini-(\d+(?:\.\d+)?)", name)
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
        raise LLMError("Semua model gagal | " + " | ".join(errors))


    # ──────────────────────────────────────────────────────────────
    # Protocol: OpenAI-compatible (OpenAI, OpenRouter, Groq, Mistral)
    # ──────────────────────────────────────────────────────────────
    def _call_openai_compat(self, model: str, system: str, user: str, temperature: float, max_tokens: int) -> str:
        base = self.base_url.rstrip("/")
        headers: dict = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        # OpenRouter butuh header tambahan
        if "openrouter.ai" in base:
            headers["HTTP-Referer"] = "https://github.com/Fahmigans1337/rag-sdm-polri"
            headers["X-Title"] = "RAG SDM Polri"

        body: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # OpenRouter: kirim daftar fallback model
        if "openrouter.ai" in base:
            p = PROVIDERS.get(self.provider_name, {})
            fbs = p.get("fallback_models", [])
            if fbs:
                body["models"] = [model] + fbs[:2]

        def _post(b: dict) -> httpx.Response:
            try:
                return httpx.post(f"{base}/chat/completions", json=b, headers=headers, timeout=self._timeout())
            except httpx.HTTPError as e:
                raise LLMError(f"Gagal menghubungi {self.provider_name}: {type(e).__name__}") from e

        r = _post(body)
        # Beberapa model baru pakai max_completion_tokens
        if r.status_code == 400 and "max_tokens" in r.text:
            body.pop("max_tokens", None)
            body["max_completion_tokens"] = max_tokens
            r = _post(body)
        # Beberapa model tidak support temperature
        if r.status_code == 400 and "temperature" in r.text:
            body.pop("temperature", None)
            r = _post(body)

        if r.status_code != 200:
            wait = {429: 300, 503: 30, 404: 3600}.get(r.status_code, 30)
            self._cooldown[model] = time.time() + wait
            raise LLMError(f"HTTP {r.status_code}: {r.text[:200]}".replace("\n", " "))
        try:
            return r.json()["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as e:
            raise LLMError(f"Respons tidak terduga: {r.text[:300]}") from e

    # ──────────────────────────────────────────────────────────────
    # Protocol: Anthropic (Claude)
    # ──────────────────────────────────────────────────────────────
    def _call_anthropic(self, model: str, system: str, user: str, temperature: float, max_tokens: int) -> str:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        body = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "temperature": temperature,
        }
        try:
            r = httpx.post("https://api.anthropic.com/v1/messages", json=body, headers=headers, timeout=self._timeout())
        except httpx.HTTPError as e:
            raise LLMError(f"Gagal menghubungi Anthropic: {type(e).__name__}") from e

        if r.status_code != 200:
            wait = {429: 300, 503: 30, 404: 3600}.get(r.status_code, 30)
            self._cooldown[model] = time.time() + wait
            raise LLMError(f"HTTP {r.status_code}: {r.text[:200]}".replace("\n", " "))
        try:
            return r.json()["content"][0]["text"].strip()
        except (KeyError, IndexError) as e:
            raise LLMError(f"Respons Anthropic tidak terduga: {r.text[:300]}") from e

    # ──────────────────────────────────────────────────────────────
    # Protocol: Google Gemini
    # ──────────────────────────────────────────────────────────────
    def _call_gemini(self, model: str, system: str, user: str, temperature: float, max_tokens: int) -> str:
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
            raise LLMError(f"HTTP {r.status_code}: {r.text[:200]}".replace("\n", " "))
        try:
            parts = r.json()["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
        except (KeyError, IndexError) as e:
            raise LLMError(f"Respons Gemini tidak terduga: {r.text[:300]}") from e


# ─── Fungsi utilitas API key dari UI ─────────────────────────────────────────

def _key_file():
    return settings.INDEX_DIR / "llm_key.json"


def _validate_key(key: str, provider_name: str) -> None:
    """Validasi key ke endpoint provider."""
    p = PROVIDERS[provider_name]
    protocol = p["protocol"]

    if protocol == "gemini":
        url = f"{p['base_url']}/models"
        headers = {"x-goog-api-key": key}
    elif protocol == "anthropic":
        url = "https://api.anthropic.com/v1/models"
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
    elif provider_name == "openrouter":
        url = "https://openrouter.ai/api/v1/key"  # /models bersifat publik, /key butuh auth
        headers = {"Authorization": f"Bearer {key}"}
    else:
        url = f"{p['base_url']}/models"
        headers = {"Authorization": f"Bearer {key}"}

    try:
        r = httpx.get(url, headers=headers, timeout=15)
    except httpx.HTTPError as e:
        raise LLMError(f"Tidak dapat menghubungi {p['name']} ({type(e).__name__}). Periksa koneksi.") from e

    # Gemini mengembalikan 400 (API key not valid) untuk key salah
    if r.status_code in (401, 403) or (protocol == "gemini" and r.status_code == 400):
        raise LLMError(f"API key ditolak oleh {p['name']} (HTTP {r.status_code}). Periksa kembali key Anda.")
    if r.status_code >= 500:
        raise LLMError(f"{p['name']} sedang bermasalah (HTTP {r.status_code}). Coba lagi sebentar.")


def apply_key(key: str, validate: bool = True, provider_hint: str = "", target: "LLM | None" = None) -> str:
    """Pasang API key saat runtime (dari UI). Provider dideteksi otomatis dari prefix key.
    Memvalidasi ke penyedia (kecuali validate=False), menyimpan ke volume, memasang ke instance `target`."""
    key = re.sub(r"\s", "", key or "")
    if len(key) < 12 or not key.isascii():
        raise LLMError("Format API key tidak valid.")

    pname = provider_hint or detect_provider(key)
    if pname and pname not in PROVIDERS:
        pname = None

    if pname is None:
        # Format key tidak dikenali dari awalan: coba ke tiap provider, pakai yang menerima.
        # (validate=False dipakai saat memuat key tersimpan, yang selalu menyertakan provider.)
        if not validate:
            raise LLMError("Provider key tersimpan tidak diketahui.")
        tried: list[str] = []
        for cand in ("gemini", "openrouter", "openai", "groq", "anthropic", "mistral"):
            try:
                _validate_key(key, cand)
                pname = cand
                break
            except LLMError as e:
                tried.append(PROVIDERS[cand]["name"])
                if "tidak dapat menghubungi" in str(e).lower():
                    raise  # masalah jaringan, bukan masalah key
        if pname is None:
            raise LLMError(
                "API key ditolak oleh semua penyedia yang didukung (" + ", ".join(tried) + "). "
                "Periksa apakah key lengkap, masih aktif, dan belum dicabut."
            )
    elif validate:
        _validate_key(key, pname)

    p = PROVIDERS[pname]
    model = p["default_model"]
    if target is not None:
        target._set_provider(pname, key, model)

    try:
        settings.INDEX_DIR.mkdir(parents=True, exist_ok=True)
        f = _key_file()
        f.write_text(json.dumps({"key": key, "provider": pname, "model": model}), encoding="utf-8")
        f.chmod(0o600)
    except OSError as e:
        log.warning("Key tidak dapat disimpan ke disk (%s); hanya aktif sampai restart.", e)

    return f"API key {p['name']} aktif · model {model}"


def load_saved_key(target: "LLM | None" = None) -> bool:
    """Muat key yang disimpan dari UI (jika ada). Dipanggil saat startup."""
    f = _key_file()
    try:
        if f.is_file():
            data = json.loads(f.read_text(encoding="utf-8"))
            apply_key(data["key"], validate=False, provider_hint=data.get("provider", ""), target=target)
            log.info("API key dimuat dari penyimpanan (provider=%s).", data.get("provider", "auto"))
            return True
    except (OSError, ValueError, KeyError, LLMError) as e:
        log.warning("Key tersimpan tidak dapat dimuat: %s", e)
    return False
