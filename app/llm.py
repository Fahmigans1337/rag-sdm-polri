"""Penyedia LLM: Gemini, OpenAI-compatible (OpenAI/Groq/OpenRouter/Ollama), atau tanpa LLM."""
from __future__ import annotations

import logging
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
        if self.provider == "gemini":
            return self._gemini(system, user, temperature, max_tokens)
        if self.provider == "openai":
            return self._openai(system, user, temperature, max_tokens)
        raise LLMError("LLM tidak dikonfigurasi")

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
        raise LLMError("Semua model Gemini gagal | " + " | ".join(errors))

    # -- OpenAI-compatible
    def _openai(self, system: str, user: str, temperature: float, max_tokens: int) -> str:
        base = (settings.OPENAI_BASE_URL or "https://api.openai.com/v1").rstrip("/")
        headers = {"Authorization": f"Bearer {settings.OPENAI_API_KEY or 'ollama'}"}
        body = {
            "model": settings.OPENAI_MODEL,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            r = httpx.post(f"{base}/chat/completions", json=body, headers=headers, timeout=settings.LLM_TIMEOUT)
        except httpx.HTTPError as e:
            raise LLMError(f"Gagal menghubungi LLM: {e}") from e
        if r.status_code != 200:
            raise LLMError(f"LLM HTTP {r.status_code}: {r.text[:300]}")
        try:
            return r.json()["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as e:
            raise LLMError(f"Respons LLM tidak terduga: {r.text[:300]}") from e
