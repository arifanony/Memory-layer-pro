"""Same-endpoint model fallback chain (LLMCLIENT).

Fallbacks trigger only when no model response was obtained (transport or
provider-envelope failure). Malformed model output never triggers a retry:
it is reported, not hidden behind another call.
"""

import io
import json
import urllib.error
import urllib.request

from config import load, validate
from llm_client import LLMError, build_request, complete


def _ok_payload(answer="done"):
    return {"answer": answer, "memory_proposals": []}


def _http(payload):
    body = json.dumps(
        {"choices": [{"message": {"content": json.dumps(payload)}}]}).encode()
    return io.BytesIO(body)


def _settings(**overrides):
    base = {"NVIDIA_API_KEY": "k", "NVIDIA_MODEL": "primary",
            "LLM_FALLBACK_MODELS": "fallback-a, fallback-b"}
    base.update(overrides)
    return validate(load(base))


def test_fallback_models_parsed_and_deduped():
    settings = load({"NVIDIA_API_KEY": "k", "NVIDIA_MODEL": "primary",
                     "LLM_FALLBACK_MODELS": " fallback-a,,fallback-a , fallback-b "})
    assert settings.llm_fallback_models == ("fallback-a", "fallback-b")


def test_complete_retries_next_model_on_transport_failure(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout=60):
        calls.append(json.loads(request.data.decode())["model"])
        if len(calls) == 1:
            raise urllib.error.URLError("rate limited")
        return _http(_ok_payload())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = complete([{"role": "user", "content": "hi"}], _settings())
    assert result.answer == "done"
    assert result.model == "fallback-a"
    assert calls == ["primary", "fallback-a"]


def test_complete_does_not_retry_on_malformed_output(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout=60):
        calls.append(request)
        return _http({"answer": "  ", "memory_proposals": []})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    try:
        complete([{"role": "user", "content": "hi"}], _settings())
    except LLMError:
        pass
    else:
        raise AssertionError("malformed output must still raise")
    assert len(calls) == 1


def test_null_content_retries_next_model(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout=60):
        calls.append(json.loads(request.data.decode())["model"])
        if len(calls) == 1:
            return io.BytesIO(json.dumps(
                {"choices": [{"message": {"content": None},
                              "finish_reason": "length"}]}).encode())
        return _http(_ok_payload())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = complete([{"role": "user", "content": "hi"}], _settings())
    assert result.answer == "done"
    assert result.model == "fallback-a"
    assert calls == ["primary", "fallback-a"]


def test_null_content_on_all_models_names_them(monkeypatch):
    def fake_urlopen(request, timeout=60):
        return io.BytesIO(json.dumps(
            {"choices": [{"message": {"content": None},
                          "finish_reason": "length"}]}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    try:
        complete([{"role": "user", "content": "hi"}], _settings())
    except LLMError as exc:
        message = str(exc)
    else:
        raise AssertionError("empty content on every model must raise")
    assert "primary" in message and "fallback-a" in message


def test_complete_failure_names_models_without_leaking_key(monkeypatch):
    def boom(request, timeout=60):
        raise urllib.error.URLError("down")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    try:
        complete([{"role": "user", "content": "hi"}], _settings())
    except LLMError as exc:
        message = str(exc)
    else:
        raise AssertionError("total outage must raise")
    assert "primary" in message and "fallback-a" in message
    assert "sk-" not in message and ": k" not in message and "Bearer" not in message


def test_build_request_model_override():
    request = build_request([{"role": "user", "content": "hi"}],
                            _settings(), model="fallback-b")
    assert json.loads(request.data.decode())["model"] == "fallback-b"
