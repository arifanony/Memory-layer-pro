"""One-call hosted chat client (LLMCLIENT).

Exactly one HTTP request per successful turn. The model is asked for a JSON
object with an answer plus memory proposals; everything else — screening,
policy, persistence — happens in Python elsewhere. Parsing is strict: any
deviation is an LLMError, never a silent second call.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, replace

from config import Settings, active_api_key, active_base_url, active_model
from schemas import CATEGORIES, RESULT_SCHEMA


class LLMError(Exception):
    """Raised for transport failures and malformed model output. User-facing."""


class LLMTransportError(LLMError):
    """No model response was obtained (connection, HTTP, or envelope failure).

    Only this class triggers a fallback-model retry. Malformed model output
    stays a plain LLMError so it is reported, never hidden behind another call.
    """


@dataclass(frozen=True)
class Proposal:
    text: str
    category: str
    confidence: float | None = None


@dataclass(frozen=True)
class ModelResult:
    answer: str
    proposals: tuple[Proposal, ...]
    model: str = ""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise LLMError(f"Invalid model output: {message}")


def parse_result(payload: object) -> ModelResult:
    """Validate a decoded JSON value against RESULT_SCHEMA. Strict, no repair."""
    _check(isinstance(payload, dict), "top-level value must be an object")
    assert isinstance(payload, dict)
    answer = payload.get("answer")
    _check(isinstance(answer, str) and len(answer.strip()) > 0, "answer must be non-empty")
    assert isinstance(answer, str)
    _check(len(answer) <= RESULT_SCHEMA["properties"]["answer"]["maxLength"],
           f"answer exceeds {RESULT_SCHEMA['properties']['answer']['maxLength']} chars")
    raw_proposals = payload.get("memory_proposals")
    _check(isinstance(raw_proposals, list), "memory_proposals must be an array")
    assert isinstance(raw_proposals, list)
    item_schema = RESULT_SCHEMA["properties"]["memory_proposals"]
    _check(len(raw_proposals) <= item_schema["maxItems"],
           f"at most {item_schema['maxItems']} proposals allowed")
    proposals = []
    text_schema = item_schema["items"]["properties"]["text"]
    for item in raw_proposals:
        _check(isinstance(item, dict), "each proposal must be an object")
        assert isinstance(item, dict)
        text = item.get("text")
        _check(isinstance(text, str) and text_schema["minLength"] <= len(text) <= text_schema["maxLength"],
               "proposal text must be 1-500 chars")
        assert isinstance(text, str)
        _check(item.get("category") in CATEGORIES,
               f"proposal category must be one of {list(CATEGORIES)}")
        confidence = item.get("confidence")
        if confidence is not None:
            _check(isinstance(confidence, (int, float)) and 0.0 <= confidence <= 1.0,
                   "confidence must be between 0 and 1")
        proposals.append(Proposal(text=text, category=item["category"],
                                  confidence=confidence))
    return ModelResult(answer=answer, proposals=tuple(proposals))


def build_request(messages: list[dict], settings: Settings,
                  model: str | None = None) -> urllib.request.Request:
    """One chat-completions request: JSON mode, deterministic, bounded."""
    body = {
        "model": model or active_model(settings),
        "messages": messages,
        "temperature": 0,
        "max_tokens": 2000,
        "response_format": {"type": "json_object"},
    }
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {active_api_key(settings)}"}
    if settings.llm_provider == "openrouter":
        # OpenRouter ranking/analytics headers; harmless to the API itself.
        headers["HTTP-Referer"] = "http://localhost:8501"
        headers["X-Title"] = "Memory Layer Pro"
    return urllib.request.Request(
        active_base_url(settings).rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )


def complete(messages: list[dict], settings: Settings,
            timeout: int | None = None) -> ModelResult:
    """POST and parse strictly. Transport failures walk the fallback chain;
    malformed model output raises immediately and never triggers a retry."""
    if timeout is None:
        timeout = settings.llm_timeout_seconds
    models: list[str] = []
    for candidate in (active_model(settings), *settings.llm_fallback_models):
        if candidate and candidate not in models:
            models.append(candidate)
    failures: list[str] = []
    for model in models:
        try:
            return replace(_attempt(messages, settings, model, timeout), model=model)
        except LLMTransportError as exc:
            failures.append(f"{model} ({exc})"[:200])
    raise LLMError(
        f"Hosted call failed on all {len(models)} model(s): " + "; ".join(failures))


def _attempt(messages: list[dict], settings: Settings,
             model: str, timeout: int) -> ModelResult:
    """One POST plus strict parse. Transport problems are LLMTransportError;
    anything wrong with an obtained model response stays a plain LLMError."""
    try:
        with urllib.request.urlopen(build_request(messages, settings, model=model),
                                    timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise LLMTransportError(f"Hosted call failed: {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMTransportError(f"Model did not return JSON: {exc}") from exc
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"Unexpected response envelope: {exc}") from exc
    if content is None:
        # Reasoning models can spend the whole token budget thinking and
        # return no message content (finish_reason "length"). Nothing usable
        # was obtained, so this is a transport-class failure: the fallback
        # chain gets a turn instead of failing the user's turn outright.
        raise LLMTransportError("Model returned no content (empty message).")
    try:
        inner = json.loads(content)
    except (json.JSONDecodeError, TypeError) as exc:
        raise LLMError(f"Message content is not JSON: {exc}") from exc
    return parse_result(inner)
