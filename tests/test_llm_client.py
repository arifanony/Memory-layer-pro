"""One-call client with strict parsing (LLMCLIENT; FR-6, NFR-1, AC-3)."""

import io
import json
import urllib.error
import urllib.request

import pytest

from config import validate, load
from llm_client import LLMError, build_request, complete, parse_result


def _settings():
    return validate(load({"NVIDIA_API_KEY": "k", "NVIDIA_MODEL": "m"}))


def _result(answer="do this", proposals=()):
    return {"answer": answer,
            "memory_proposals": [{"text": t, "category": c} for t, c in proposals]}


def _http(payload):
    body = json.dumps({"choices": [{"message": {"content": json.dumps(payload)}}]}).encode()
    return io.BytesIO(body)


def test_parse_valid_result():
    result = parse_result(_result("hi", [("likes tea", "preference")]))
    assert result.answer == "hi"
    assert result.proposals[0].category == "preference"


@pytest.mark.parametrize("bad", [
    {},
    {"answer": "  ", "memory_proposals": []},
    {"answer": "x" * 4001, "memory_proposals": []},
    {"answer": "ok", "memory_proposals": "nope"},
    {"answer": "ok", "memory_proposals": [{"text": "t", "category": "fact"}] * 6},
    {"answer": "ok", "memory_proposals": [{"text": "", "category": "fact"}]},
    {"answer": "ok", "memory_proposals": [{"text": "t", "category": "mood"}]},
    {"answer": "ok", "memory_proposals": [{"text": "t", "category": "fact", "confidence": 2}]},
])
def test_parse_rejects_malformed(bad):
    with pytest.raises(LLMError):
        parse_result(bad)


def test_complete_posts_once_with_monkeypatched_http(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout=60):
        calls.append((request, timeout))
        return _http(_result("done"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = complete([{"role": "user", "content": "hi"}], _settings())
    assert result.answer == "done"
    assert len(calls) == 1
    sent = json.loads(calls[0][0].data.decode())
    assert sent["temperature"] == 0 and sent["response_format"] == {"type": "json_object"}


def test_complete_uses_settings_timeout_by_default(monkeypatch):
    seen = []

    def fake_urlopen(request, timeout=60):
        seen.append(timeout)
        return _http(_result("done"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    complete([{"role": "user", "content": "hi"}], _settings())
    assert seen == [120]


def test_complete_raises_on_transport_error(monkeypatch):
    def boom(request, timeout=60):
        raise urllib.error.URLError("down")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(LLMError, match="[Hh]osted call failed"):
        complete([{"role": "user", "content": "hi"}], _settings())


def test_complete_raises_on_non_json_envelope(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda request, timeout=60: io.BytesIO(b"not json"))
    with pytest.raises(LLMError):
        complete([{"role": "user", "content": "hi"}], _settings())
