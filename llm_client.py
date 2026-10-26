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
from dataclasses import dataclass

from config import Settings
from schemas import CATEGORIES, RESULT_SCHEMA


class LLMError(Exception):
    """Raised for transport failures and malformed model output. User-facing."""


@dataclass(frozen=True)
class Proposal:
    text: str
    category: str
    confidence: float | None = None


@dataclass(frozen=True)
class ModelResult:
    answer: str
    proposals: tuple[Proposal, ...]


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


def build_request(messages: list[dict], settings: Settings) -> urllib.request.Request:
    """One chat-completions request: JSON mode, deterministic, bounded."""
    body = {
        "model": settings.nvidia_model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 2000,
        "response_format": {"type": "json_object"},
    }
    return urllib.request.Request(
        settings.nvidia_base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {settings.nvidia_api_key}"},
        method="POST",
    )


def complete(messages: list[dict], settings: Settings, timeout: int = 60) -> ModelResult:
    """POST once and parse strictly. Any failure is LLMError, never a retry."""
    try:
        with urllib.request.urlopen(build_request(messages, settings),
                                    timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise LLMError(f"Hosted call failed: {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMError(f"Model did not return JSON: {exc}") from exc
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"Unexpected response envelope: {exc}") from exc
    try:
        inner = json.loads(content)
    except (json.JSONDecodeError, TypeError) as exc:
        raise LLMError(f"Message content is not JSON: {exc}") from exc
    return parse_result(inner)
