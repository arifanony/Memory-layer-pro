"""Shared contracts for Memory Layer Pro (SCAFFOLD).

Schemas live here so the LLM client, orchestrator, and UI validate against
one definition. Logic arrives with LLMCLIENT/ORCHESTRATOR; this module only
pins the shapes the ADR specifies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

# Allowlisted memory categories. Ephemeral statements are filtered by policy,
# never persisted; the rest map to durable semantic/procedural memory.
CATEGORIES = ("fact", "preference", "procedure")


@dataclass(frozen=True)
class StoredMemory:
    id: str
    text: str
    category: str
    score: float = 0.0


@dataclass
class MemoryEvent:
    type: str  # ADD, UPDATE, SUPERSEDE, REJECT
    reason: str
    memory_id: str | None = None
    supersedes: str | None = None


class MemoryStore(Protocol):
    """Durable backend contract. MEMSTORE implements this on pgvector."""

    def search(self, user_id: str, vector: list[float], top_k: int) -> list[StoredMemory]: ...
    def add(self, user_id: str, text: str, category: str,
            vector: list[float]) -> StoredMemory: ...
    def link_supersede(self, old_id: str, new_id: str) -> None: ...
    def record_event(self, user_id: str, event: MemoryEvent) -> None: ...


class WorkingMemory(Protocol):
    """Short-lived context contract. WORKMEM implements this on Redis.

    Keys are scoped to one user and one session (`wm:{user_id}:{session_id}`)
    so demo users never share working context.
    """

    def read(self, user_id: str, session_id: str, limit: int) -> list[str]: ...
    def append(self, user_id: str, session_id: str, turn: str) -> None: ...
    def clear(self, user_id: str, session_id: str) -> None: ...
    def clear_user(self, user_id: str) -> int: ...

# JSON contract for one hosted model result. The model returns an answer plus
# zero or more memory proposals; Python validates and decides persistence.
RESULT_SCHEMA = {
    "type": "object",
    "required": ["answer", "memory_proposals"],
    "properties": {
        "answer": {"type": "string", "minLength": 1, "maxLength": 4000},
        "memory_proposals": {
            "type": "array",
            "maxItems": 5,
            "items": {
                "type": "object",
                "required": ["text", "category"],
                "properties": {
                    "text": {"type": "string", "minLength": 1, "maxLength": 500},
                    "category": {"type": "string", "enum": list(CATEGORIES)},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
            },
        },
    },
}
