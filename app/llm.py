"""Penyedia LLM: Gemini, OpenAI, OpenRouter, Groq, Mistral, Anthropic Claude, atau ekstraktif tanpa LLM."""
from __future__ import annotations

import json
import logging
import re
import time
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

    def _discover_gemini_models(self) -> list[str]:
        """Ambil daftar model Gemini yang benar-benar tersedia untuk key ini (nama model sering diganti Google)."""
        try:
            r = httpx.get(
                f"{PROVIDERS['gemini']['base_url']}/models?pageSize=200",
                headers={"x-goog-api-key": self.api_key}, timeout=15,
            )
            if r.status_code != 200:
                return []
            cands = []
            for m in r.json().get("models", []):
                name = m.get("name", "").removeprefix("models/")
                if "generateContent" not in m.get("supportedGenerationMethods", []):
                    continue
                if not name.startswith("gemini-") or "flash" not in name:
                    continue
                if re.search(r"image|tts|live|audio|embed|thinking|exp|8b|robotics|computer|custom|learnlm", name):
                    continue
                v = re.match(r"gemini-(\d+(?:\.\d+)?)", name)
                ver = float(v.group(1)) if v else 0.0
                cands.append((-ver, "preview" in name, "lite" in name, name))
            cands.sort()
            found = [c[3] for c in cands][:4]
            log.info("Model Gemini tersedia: %s", found)
            return found
        except (httpx.HTTPError, ValueError, KeyError) as e:
            log.warning("Gagal mengambil daftar model Gemini: %s", type(e).__name__)
            return []

    def generate(self, system: str, user: str, temperature: float = 0.2, max_tokens: int = 1500) -> str:
        if not self.enabled:
            raise LLMError("LLM tidak dikonfigurasi. Tempel API key via tombol API KEY.")

        p = PROVIDERS.get(self.provider_name)
        if not p:
            raise LLMError(f"Provider tidak dikenal: {self.provider_name}")

        protocol = p["protocol"]
        static = [self.model] + [m for m in p.get("fallback_models", []) if m != self.model]
        if protocol == "gemini":
            if self._gem_models is None:
                self._gem_models = self._discover_gemini_models()
                if self._gem_models:
                    self._cooldown = {}
            models_to_try = self._gem_models + [m for m in static if m not in self._gem_models]
        else:
            models_to_try = static

        errors: list[str] = []
        skipped = 0
        for model in models_to_try:
            now = time.time()
            if self._cooldown.get(model, 0) > now:
                skipped += 1
                continue
            try:
                if protocol == "gemini":
                    out = self._call_gemini(model, system, user, temperature, max_tokens)
                elif protocol == "anthropic":
                    out = self._call_anthropic(model, system, user, temperature, max_tokens)
                else:
                    out = self._call_openai_compat(model, system, user, temperature, max_tokens)
                self.model = model  # update ke model yang berhasil
                self._last_model = model
                return out
            except LLMError as e:
                log.warning("Model %s gagal: %s", model, str(e)[:200])
                errors.append(f"{model}: {e}")
                # Kalau error auth, tidak perlu coba model lain
                if any(code in str(e) for code in ("HTTP 401", "HTTP 403")):
                    break
        if protocol == "gemini" and self._gem_models is not None and errors and all("HTTP 404" in e for e in errors):
            self._gem_models = None  # semua model 404: daftar model berubah, temukan ulang pada panggilan berikutnya
        if not errors and skipped:
            raise LLMError("Semua model sedang dijeda sementara setelah gagal berulang; coba lagi sebentar.")
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
                return httpx.post(f"{base}/chat/completions", json=b, headers=headers, timeout=settings.LLM_TIMEOUT)
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
            r = httpx.post("https://api.anthropic.com/v1/messages", json=body, headers=headers, timeout=settings.LLM_TIMEOUT)
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
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        for attempt in range(2):
            try:
                r = httpx.post(url, json=body, headers={"x-goog-api-key": self.api_key}, timeout=settings.LLM_TIMEOUT)
            except httpx.HTTPError as e:
                self._cooldown[model] = time.time() + 30
                raise LLMError(f"Gagal menghubungi Gemini: {type(e).__name__}") from e
            if r.status_code == 503 and attempt == 0:
                time.sleep(1.5)
                continue
            break
        if r.status_code != 200:
            wait = {429: 300, 503: 30, 404: 3600}.get(r.status_code, 30)
            self._cooldown[model] = time.time() + wait
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
