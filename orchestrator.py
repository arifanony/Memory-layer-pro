"""Turn pipeline for Memory Layer Pro (SCAFFOLD).

The 12-stage sequence below is the ADR contract. Stage logic arrives with
PRIVACY/EMBEDDINGS/MEMSTORE/WORKMEM/LLMCLIENT/ORCHESTRATOR; this module pins
the order and names so the UI trace has stable step ids from day one.
"""

from __future__ import annotations

# (step id, short label) in execution order. See ADR.md "Turn pipeline".
STAGES = (
    ("user", "select demo user/session"),
    ("screen_input", "PII/secret screen on input"),
    ("context", "read bounded Redis context"),
    ("embed", "embed protected query locally"),
    ("retrieve", "top-k pgvector retrieval, user-scoped"),
    ("prompt", "build bounded prompt"),
    ("model_call", "one Nemotron call: answer + proposals"),
    ("validate", "parse, screen, apply policy"),
    ("persist", "embed + write accepted memories"),
    ("working", "append turns to Redis per TTL"),
    ("trace", "return answer + complete trace"),
)
