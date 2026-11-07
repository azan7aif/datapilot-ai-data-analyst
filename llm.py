"""Thin LLM provider layer so the agent works with a FREE Gemini key or a paid Claude key."""
from __future__ import annotations

import os
import time

import requests

# Gemini model names change often. If the first one is unavailable (HTTP 404) the next is tried.
# fastest / most free-tier friendly first; the next one is tried if a model is missing (404) or too slow
GEMINI_MODELS = ["gemini-3.1-flash-lite", "gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-2.5-flash"]
CLAUDE_DEFAULT = os.getenv("ANALYST_MODEL", "claude-sonnet-5-5")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
RETRYABLE = {429, 500, 502, 503, 504}


class LLMError(Exception):
    """Friendly, user-facing LLM/API error."""


class AnthropicClient:
    provider = "claude"

    def __init__(self, api_key: str, model: str | None = None):
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model or CLAUDE_DEFAULT

    def complete(self, system: str, messages: list[dict], max_tokens: int = 2000) -> str:
        try:
            resp = self._client.messages.create(model=self.model, max_tokens=max_tokens,
                                                system=system, messages=messages)
        except Exception as e:  # noqa: BLE001
            raise LLMError(f"Claude API error: {e}") from e
        return "".join(getattr(b, "text", "") for b in resp.content)


class GeminiClient:
    provider = "gemini"

    def __init__(self, api_key: str, model: str | None = None):
        self.api_key = api_key
        self.model = model or GEMINI_MODELS[0]
        self.used_model = ""   # the model that actually answered (may differ if a fallback happened)

    def _candidates(self) -> list[str]:
        return [self.model] + [m for m in GEMINI_MODELS if m != self.model]

    @staticmethod
    def _friendly(r: requests.Response) -> str:
        try:
            msg = r.json().get("error", {}).get("message", r.text)
        except Exception:  # noqa: BLE001
            msg = r.text
        if r.status_code in (400, 401, 403) and ("API key" in msg or "api key" in msg or r.status_code in (401, 403)):
            return f"Gemini API key is invalid or not allowed ({r.status_code}). Create a new key at aistudio.google.com/app/apikey. Details: {msg[:200]}"
        if r.status_code == 429:
            return "Gemini free-tier limit reached. Wait a minute and try again (or try the next day if the daily quota is over)."
        return f"Gemini API error {r.status_code}: {msg[:300]}"

    def complete(self, system: str, messages: list[dict], max_tokens: int = 2000) -> str:
        contents = [{"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]}
                    for m in messages]
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": contents,
            # thinking models spend part of the budget on reasoning, so keep it generous
            "generationConfig": {"maxOutputTokens": max(max_tokens, 4096), "temperature": 0.2},
        }
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}

        seen: set[int] = set()   # HTTP statuses that made us switch model
        timeouts = 0
        for model in self._candidates():
            r = None
            for attempt in range(2):
                try:
                    r = requests.post(GEMINI_URL.format(model=model), headers=headers, json=body, timeout=(10, 60))
                except requests.Timeout:
                    r = None
                    break
                except requests.RequestException as e:
                    raise LLMError(f"Network error while calling Gemini: {e}") from e
                if r.status_code == 200:
                    self.used_model = model
                    return self._parse(r.json())
                if r.status_code in RETRYABLE:      # busy / quota: one quick retry, then switch model
                    seen.add(r.status_code)
                    if attempt == 0:
                        time.sleep(2)
                        continue
                break
            if r is None:                            # timed out -> try one more model, then give up
                timeouts += 1
                if timeouts >= 2:
                    break
                continue
            if r.status_code == 404 or r.status_code in RETRYABLE:
                continue                             # model missing or busy -> next model
            raise LLMError(self._friendly(r))        # bad key, bad request, ... (switching won't help)

        tried = ", ".join(self._candidates())
        if seen - {429}:
            raise LLMError(f"All Gemini models are busy right now (high demand, tried: {tried}). Please try again in a minute.")
        if 429 in seen:
            raise LLMError("Gemini free-tier limit reached on every model. Wait a few minutes (or until tomorrow if the daily quota is over).")
        if timeouts:
            raise LLMError("Gemini is responding too slowly right now (request timed out). Wait a minute and try again, "
                           "and check your internet connection.")
        raise LLMError("None of the Gemini models were found. Open aistudio.google.com, check the "
                       "available model names and set GEMINI_MODEL in secrets.toml.")

    @staticmethod
    def _parse(data: dict) -> str:
        cands = data.get("candidates") or []
        if not cands:
            raise LLMError(f"Gemini returned no answer (possibly blocked): {data.get('promptFeedback', '')}")
        parts = cands[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        if not text.strip():
            raise LLMError(f"Gemini returned an empty answer (finish reason: {cands[0].get('finishReason')}).")
        return text


def make_client(provider: str, api_key: str, model: str | None = None):
    if provider == "claude":
        return AnthropicClient(api_key, model or None)
    return GeminiClient(api_key, model or None)
