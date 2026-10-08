"""Penyedia LLM: Gemini, OpenAI-compatible (OpenAI/Groq/OpenRouter/Ollama), atau tanpa LLM."""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass

import httpx

from .config import settings

log = logging.getLogger("rag.llm")


class LLMError(RuntimeError):
    pass


@dataclass
class LLMInfo:
    provider: str
    model: str


class LLM:
    """Antarmuka tunggal: generate(system, user) -> str."""

    def __init__(self) -> None:
        self.provider = self._resolve()
        self._cooldown: dict[str, float] = {}  # model -> waktu (epoch) boleh dicoba lagi
        self.model = {
            "gemini": settings.GEMINI_MODEL,
            "openai": settings.OPENAI_MODEL,
        }.get(self.provider, "extractive")

    @staticmethod
    def _resolve() -> str:
        p = settings.LLM_PROVIDER
        if p in {"gemini", "openai", "extractive"}:
            return p
        if p == "none":
            return "extractive"
        if settings.GEMINI_API_KEY:
            return "gemini"
        if settings.OPENAI_API_KEY or settings.OPENAI_BASE_URL:
            return "openai"
        return "extractive"

    @property
    def enabled(self) -> bool:
        return self.provider in {"gemini", "openai"}

    @property
    def info(self) -> LLMInfo:
        return LLMInfo(self.provider, self.model)

    def generate(self, system: str, user: str, temperature: float = 0.2, max_tokens: int = 1500) -> str:
        if self.provider not in {"gemini", "openai"}:
            raise LLMError("LLM tidak dikonfigurasi")
        # Urutan percobaan: penyedia utama dulu, lalu penyedia lain yang key-nya terisi (failover)
        order = [self.provider]
        if settings.LLM_PROVIDER in {"auto", "openai", "gemini"}:
            if self.provider != "gemini" and settings.GEMINI_API_KEY:
                order.append("gemini")
            if self.provider != "openai" and settings.OPENAI_API_KEY:
                order.append("openai")
        errors: list[str] = []
        for prov in order:
            try:
                if prov == "gemini":
                    out = self._gemini(system, user, temperature, max_tokens)
                else:
                    out = self._openai(system, user, temperature, max_tokens)
                    self._last_model = settings.OPENAI_MODEL
                self._last_provider = prov
                return out
            except LLMError as e:
                log.warning("Penyedia %s gagal: %s", prov, str(e)[:200])
                errors.append(f"{prov}: {e}")
        raise LLMError(" || ".join(errors))

    def _gemini_models(self) -> list[str]:
        chain = [settings.GEMINI_MODEL] + [m.strip() for m in settings.GEMINI_FALLBACK_MODELS.split(",") if m.strip()]
        seen: set[str] = set()
        chain = [m for m in chain if not (m in seen or seen.add(m))]
        now = time.time()
        ready = [m for m in chain if self._cooldown.get(m, 0) <= now]
        return ready or chain  # jika semuanya cooldown, coba semua lagi

    def _gemini_call(self, model: str, system: str, user: str, temperature: float, max_tokens: int) -> str:
        url = f"{settings.GEMINI_BASE_URL}/models/{model}:generateContent"
        # thinking dibuat minimal agar cepat; format berbeda antar generasi model
        thinking_opts: list[dict | None] = (
            [{"thinkingBudget": 0}, None] if model.startswith("gemini-2.") else [{"thinkingLevel": "low"}, None]
        )
        last = ""
        for think in thinking_opts:
            cfg: dict = {"temperature": temperature, "maxOutputTokens": max_tokens}
            if think is not None:
                cfg["thinkingConfig"] = think
            body = {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": cfg,
            }
            r = None
            for attempt in range(2):  # sekali ulang bila model sedang sibuk (503)
                try:
                    r = httpx.post(
                        url, json=body, headers={"x-goog-api-key": settings.GEMINI_API_KEY},
                        timeout=settings.LLM_TIMEOUT,
                    )
                except httpx.HTTPError as e:
                    self._cooldown[model] = time.time() + 30
                    raise LLMError(f"Gagal menghubungi Gemini: {type(e).__name__}") from e
                if r.status_code == 503 and attempt == 0:
                    time.sleep(1.5)
                    continue
                break
            assert r is not None
            if r.status_code == 400 and think is not None:  # opsi thinking tidak didukung -> ulangi tanpa itu
                last = r.text[:200]
                continue
            if r.status_code != 200:
                # kuota habis -> istirahatkan model 5 menit; sibuk -> 30 dtk; tidak tersedia -> 1 jam
                wait = {429: 300, 503: 30, 404: 3600}.get(r.status_code, 30)
                self._cooldown[model] = time.time() + wait
                raise LLMError(f"HTTP {r.status_code}")
            data = r.json()
            try:
                parts = data["candidates"][0]["content"]["parts"]
                return "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
            except (KeyError, IndexError) as e:
                raise LLMError(f"Respons tidak terduga: {str(data)[:200]}") from e
        raise LLMError(f"HTTP 400: {last}")

    def _gemini(self, system: str, user: str, temperature: float, max_tokens: int) -> str:
        if not settings.GEMINI_API_KEY:
            raise LLMError("GEMINI_API_KEY belum diisi")
        errors: list[str] = []
        for model in self._gemini_models():
            try:
                text = self._gemini_call(model, system, user, temperature, max_tokens)
                if text:
                    self.model = model       # model yang benar-benar dipakai
                    self._last_model = model  # untuk logging
                    return text
                errors.append(f"{model}: respons kosong")
            except LLMError as e:
                log.warning("Gemini %s gagal: %s", model, e)
                errors.append(f"{model}: {e}")
                if any(code in str(e) for code in ("HTTP 401", "HTTP 403")):
                    break  # key tidak valid -> model lain pasti gagal juga
        raise LLMError("Semua model Gemini gagal | " + " | ".join(errors))

    # -- OpenAI-compatible
    def _openai(self, system: str, user: str, temperature: float, max_tokens: int) -> str:
        base = (settings.OPENAI_BASE_URL or "https://api.openai.com/v1").rstrip("/")
        if not settings.OPENAI_API_KEY and ("openai.com" in base or "openrouter.ai" in base or "groq.com" in base):
            raise LLMError("API key belum diisi (OPENAI_API_KEY kosong)")
        headers = {"Authorization": f"Bearer {settings.OPENAI_API_KEY or 'ollama'}"}
        is_openrouter = "openrouter.ai" in base
        if is_openrouter:
            headers["HTTP-Referer"] = "https://github.com/Fahmigans1337/rag-sdm-polri"
            headers["X-Title"] = "RAG SDM Polri"
        body = {
            "model": settings.OPENAI_MODEL,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if is_openrouter:
            fb = [m.strip() for m in settings.OPENAI_FALLBACK_MODELS.split(",") if m.strip()]
            if fb:  # OpenRouter mencoba model berikutnya otomatis bila yang pertama gagal
                body["models"] = [settings.OPENAI_MODEL] + fb[:2]

        def _post(b: dict) -> httpx.Response:
            try:
                return httpx.post(f"{base}/chat/completions", json=b, headers=headers, timeout=settings.LLM_TIMEOUT)
            except httpx.HTTPError as e:
                raise LLMError(f"Gagal menghubungi LLM: {type(e).__name__}") from e

        r = _post(body)
        if r.status_code == 400 and "max_tokens" in r.text:  # model baru (gpt-5/o-series) memakai max_completion_tokens
            body.pop("max_tokens", None)
            body["max_completion_tokens"] = max_tokens
            r = _post(body)
        if r.status_code == 400 and "temperature" in r.text:  # sebagian model hanya mendukung temperature default
            body.pop("temperature", None)
            r = _post(body)
        if r.status_code != 200:
            kind = "kuota/kredit habis" if r.status_code == 429 and "quota" in r.text.lower() else ""
            raise LLMError(f"LLM HTTP {r.status_code} {kind}: {r.text[:200]}".replace("\n", " "))
        try:
            return r.json()["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as e:
            raise LLMError(f"Respons LLM tidak terduga: {r.text[:300]}") from e


# --------------------------------------------------------------------------- API key dari UI
def _key_file():
    return settings.INDEX_DIR / "llm_key.json"


def _check_url(key: str) -> str:
    if key.startswith("sk-or-"):
        return "https://openrouter.ai/api/v1/key"
    if key.startswith("gsk_"):
        return "https://api.groq.com/openai/v1/models"
    return "https://api.openai.com/v1/models"


def apply_key(key: str, validate: bool = True) -> str:
    """Pasang API key saat runtime (dari UI). Jenis penyedia dikenali dari awalan key.
    Memvalidasi ke penyedia (kecuali validate=False), menyimpan ke volume, dan mengembalikan pesan sukses."""
    from .config import _provider_defaults  # lokal agar tidak menambah dependensi impor

    key = re.sub(r"\s", "", key or "")
    if len(key) < 12 or not key.isascii():
        raise LLMError("Format API key tidak valid.")
    if validate:
        try:
            r = httpx.get(_check_url(key), headers={"Authorization": f"Bearer {key}"}, timeout=15)
        except httpx.HTTPError as e:
            raise LLMError(f"Tidak dapat menghubungi penyedia API ({type(e).__name__}). Periksa koneksi internet.") from e
        if r.status_code in (401, 403):
            raise LLMError(f"API key ditolak oleh penyedia (HTTP {r.status_code}). Periksa kembali key Anda.")
        if r.status_code >= 500:
            raise LLMError(f"Penyedia API sedang bermasalah (HTTP {r.status_code}). Coba lagi sebentar.")
    d = _provider_defaults(key)
    settings.OPENAI_API_KEY = key
    settings.OPENAI_BASE_URL = d["base"]
    settings.OPENAI_MODEL = d["model"]
    settings.OPENAI_FALLBACK_MODELS = d["fallback"]
    settings.LLM_PROVIDER = "openai"
    try:  # simpan agar tetap aktif setelah container restart (volume rag_index)
        settings.INDEX_DIR.mkdir(parents=True, exist_ok=True)
        f = _key_file()
        f.write_text(json.dumps({"key": key}), encoding="utf-8")
        f.chmod(0o600)
    except OSError as e:  # noqa: BLE001
        log.warning("Key tidak dapat disimpan ke disk (%s); hanya aktif sampai restart.", e)
    name = "OpenRouter" if key.startswith("sk-or-") else "Groq" if key.startswith("gsk_") else "OpenAI"
    return f"API key {name} aktif (model {d['model']})."


def load_saved_key() -> bool:
    """Muat key yang disimpan dari UI (jika ada). Dipanggil saat startup; key dari UI menang atas env."""
    f = _key_file()
    try:
        if f.is_file():
            apply_key(json.loads(f.read_text(encoding="utf-8"))["key"], validate=False)
            log.info("API key dimuat dari penyimpanan (dipasang lewat UI).")
            return True
    except (OSError, ValueError, KeyError, LLMError) as e:
        log.warning("Key tersimpan tidak dapat dimuat: %s", e)
    return False
