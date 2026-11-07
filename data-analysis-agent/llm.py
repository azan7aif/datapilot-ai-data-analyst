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

        r = None
        timeouts = 0
        for model in self._candidates():
            for attempt in range(3):
                try:
                    r = requests.post(GEMINI_URL.format(model=model), headers=headers, json=body, timeout=(10, 60))
                except requests.Timeout:
                    r = None
                    break
                except requests.RequestException as e:
                    raise LLMError(f"Network error while calling Gemini: {e}") from e
                if r.status_code == 200:
                    self.model = model  # remember the model that worked
                    return self._parse(r.json())
                if r.status_code in (429, 503) and attempt < 2:
                    time.sleep(5 * (attempt + 1))
                    continue
                break
            if r is None:  # this model timed out -> try one more (lighter) model, then give up
                timeouts += 1
                if timeouts >= 2:
                    break
                continue
            if r.status_code != 404:
                raise LLMError(self._friendly(r))
        if timeouts:
            raise LLMError("Gemini is responding too slowly right now (request timed out). Wait a minute and try again. "
                           "Tip: use a light model such as gemini-3.1-flash-lite in the sidebar and check your internet connection.")
        raise LLMError("None of the Gemini models were found. Open aistudio.google.com, check the "
                       "available model names and type one in the sidebar 'Model' box.")

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
