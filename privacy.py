"""Local PII/secret screening (PRIVACY).

Pattern-based detector behind a small interface so a stronger implementation
can replace it later. Runs before any hosted request: `mask` redacts values
with typed placeholders, and flagged content is rejected from storage by
policy. Detection is best-effort — it misses things and false-positives —
so the app stays synthetic-data-only and never claims real protection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Finding:
    category: str
    value: str
    start: int
    end: int


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = ord(ch) - 48
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# (category, compiled pattern). Order matters: specific secrets first so a
# credential line is labeled SECRET, not split across looser patterns.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("SECRET", re.compile(
        r"(?i)\b(api[_-]?key|secret|token|password|passwd|pwd)\b\s*[:=]\s*\S+"
        r"|\bsk-[A-Za-z0-9]{8,}\b|\bghp_[A-Za-z0-9]{8,}\b"
        r"|\bxox[bap]-[A-Za-z0-9-]{8,}\b|\bAKIA[0-9A-Z]{16}\b"
    )),
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("PHONE", re.compile(
        r"(?<!\d)(?:\+\d{1,3}[-.\s]?)?(?:\(\d{2,4}\)[-.\s]?|\d{2,4}[-.\s])"
        r"\d{3,4}[-.\s]?\d{4}(?!\d)"
    )),
    ("IP", re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
        r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b"
    )),
    ("CARD", re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")),
    ("GOV_ID", re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")),
)


def scan(text: str) -> list[Finding]:
    """Return non-overlapping findings, earliest-first. CARD needs Luhn."""
    found: list[Finding] = []
    taken: list[tuple[int, int]] = []
    for category, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(start < stop and begin < end for begin, stop in taken):
                continue
            value = match.group()
            if category == "CARD" and not _luhn_ok(re.sub(r"[ -]", "", value)):
                continue
            taken.append((start, end))
            found.append(Finding(category, value, start, end))
    found.sort(key=lambda item: item.start)
    return found


def contains_pii(text: str) -> bool:
    """True when any detector fires. Cheap pre-check before masking."""
    return any(True for _ in scan(text))


@dataclass
class MaskResult:
    text: str
    mapping: dict[str, str] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)


def mask(text: str) -> MaskResult:
    """Replace findings with typed placeholders ([EMAIL_1], ...).

    The mapping is request-scoped: keep it in memory for the turn, never log
    or store it. Restoration is opt-in at the call site; default is no restore.
    """
    result = MaskResult(text=text)
    offset = 0
    for finding in scan(text):
        number = result.counts.get(finding.category, 0) + 1
        result.counts[finding.category] = number
        token = f"[{finding.category}_{number}]"
        result.mapping[token] = finding.value
        begin = finding.start + offset
        result.text = result.text[:begin] + token + result.text[finding.end + offset:]
        offset += len(token) - (finding.end - finding.start)
    return result


def restore(masked: str, mapping: dict[str, str]) -> str:
    """Put originals back for the user-facing reply only. Discard after."""
    for token, value in mapping.items():
        masked = masked.replace(token, value)
    return masked
