import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import llm
from llm import GeminiClient, LLMError, make_client


class Resp:
    def __init__(self, status, data):
        self.status_code, self._d, self.text = status, data, str(data)

    def json(self):
        return self._d


OK = {"candidates": [{"content": {"parts": [{"text": "hello"}, {"text": "hidden", "thought": True}]}}]}


def test_gemini_success_and_payload(monkeypatch):
    seen = {}
    def fake_post(url, headers, json, timeout):
        seen.update(url=url, headers=headers, body=json)
        return Resp(200, OK)
    monkeypatch.setattr(llm.requests, "post", fake_post)
    c = make_client("gemini", "KEY", "gemini-x")
    out = c.complete("sys", [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}, {"role": "user", "content": "q2"}])
    assert out == "hello"
    assert "gemini-x:generateContent" in seen["url"] and seen["headers"]["x-goog-api-key"] == "KEY"
    assert [m["role"] for m in seen["body"]["contents"]] == ["user", "model", "user"]
    assert seen["body"]["systemInstruction"]["parts"][0]["text"] == "sys"


def test_gemini_falls_back_on_404(monkeypatch):
    tried = []
    def fake_post(url, **kw):
        tried.append(url)
        return Resp(404, {"error": {"message": "not found"}}) if len(tried) == 1 else Resp(200, OK)
    monkeypatch.setattr(llm.requests, "post", fake_post)
    c = GeminiClient("K", "old-model")
    assert c.complete("s", [{"role": "user", "content": "q"}]) == "hello"
    assert len(tried) == 2 and c.model != "old-model"


def test_gemini_bad_key_message(monkeypatch):
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: Resp(400, {"error": {"message": "API key not valid"}}))
    with pytest.raises(LLMError, match="invalid"):
        GeminiClient("bad").complete("s", [{"role": "user", "content": "q"}])


def test_gemini_rate_limit_retries_then_friendly(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    calls = []
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: (calls.append(1), Resp(429, {"error": {"message": "quota"}}))[1])
    with pytest.raises(LLMError, match="limit"):
        GeminiClient("k").complete("s", [{"role": "user", "content": "q"}])
    assert len(calls) == 3


def test_empty_answer(monkeypatch):
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: Resp(200, {"candidates": []}))
    with pytest.raises(LLMError):
        GeminiClient("k").complete("s", [{"role": "user", "content": "q"}])


def test_gemini_timeout_falls_back_then_friendly(monkeypatch):
    calls = []
    def fake_post(url, **kw):
        calls.append(url)
        raise llm.requests.Timeout("slow")
    monkeypatch.setattr(llm.requests, "post", fake_post)
    with pytest.raises(LLMError, match="too slowly"):
        GeminiClient("k").complete("s", [{"role": "user", "content": "q"}])
    assert len(calls) == 2 and calls[0] != calls[1]


def test_gemini_timeout_then_success_on_next_model(monkeypatch):
    n = {"i": 0}
    def fake_post(url, **kw):
        n["i"] += 1
        if n["i"] == 1:
            raise llm.requests.Timeout("slow")
        return Resp(200, OK)
    monkeypatch.setattr(llm.requests, "post", fake_post)
    assert GeminiClient("k").complete("s", [{"role": "user", "content": "q"}]) == "hello"
