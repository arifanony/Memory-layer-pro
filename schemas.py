"""Shared contracts for Memory Layer Pro (SCAFFOLD).

Schemas live here so the LLM client, orchestrator, and UI validate against
one definition. Logic arrives with LLMCLIENT/ORCHESTRATOR; this module only
pins the shapes the ADR specifies.
"""

from __future__ import annotations

# Allowlisted memory categories. Ephemeral statements are filtered by policy,
# never persisted; the rest map to durable semantic/procedural memory.
CATEGORIES = ("fact", "preference", "procedure")

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
