"""Turn pipeline with policy enforcement and honest traces (ORCHESTRATOR).

Implements the ADR 11-stage sequence. Storage backends arrive with MEMSTORE
and WORKMEM; this module defines the narrow protocols they must satisfy and
ships in-memory fakes that the tests (and only the tests) use. Production
wiring happens in those tasks, not here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from config import Settings
from llm_client import LLMError, ModelResult, Proposal, complete
from privacy import MaskResult, contains_pii, mask
from schemas import MemoryEvent, MemoryStore, StoredMemory, WorkingMemory

SYSTEM_PROMPT = (
    "You are a memory-aware assistant. Reply with JSON only, exactly:\n"
    '{"answer": string, "memory_proposals": [{"text": string, '
    '"category": "fact" | "preference" | "procedure", "confidence": number?}]}.\n'
    "Propose a memory only for durable, user-specific facts worth remembering. "
    "Never propose secrets, contact details, or transient status updates."
)

# Same-category near-duplicate text counts as a contradiction to version,
# not a duplicate to drop. ponytail: one threshold, documented here.
SUPERSEDE_SIMILARITY = 0.9


@dataclass
class TraceStep:
    id: str
    label: str
    status: str  # ok, skipped, failed
    ms: float = 0.0
    detail: str = ""


@dataclass
class TurnResult:
    answer: str
    persisted: bool
    working_saved: bool
    memories: list[StoredMemory] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    events: list[MemoryEvent] = field(default_factory=list)
    trace: list[TraceStep] = field(default_factory=list)


def build_prompt(masked_input: str, context: list[str],
                 memories: list[StoredMemory], retrieval_on: bool) -> list[dict]:
    """Assemble the bounded prompt. Retrieval-off provably excludes memories."""
    parts = [f"User message: {masked_input}"]
    if context:
        parts.append("Recent context:\n" + "\n".join(context))
    if retrieval_on and memories:
        parts.append("Relevant memories:\n" + "\n".join(
            f"- [{mem.category}] {mem.text}" for mem in memories))
    parts.append("Answer helpfully and propose durable memories as specified.")
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(parts)}]


def _screen_memories(memories: list[StoredMemory], trace: list[TraceStep],
                      ) -> tuple[list[StoredMemory], MaskResult | None]:
    """Defense in depth: re-screen retrieved memories before prompt use."""
    safe = []
    for mem in memories:
        if contains_pii(mem.text):
            trace.append(TraceStep(id="screen_memory", label="Drop tainted memory",
                                   status="skipped",
                                   detail="retrieved memory contained PII; excluded"))
        else:
            safe.append(mem)
    return safe, None


def run_turn(user_text: str, user_id: str, session_id: str, settings: Settings,
             store: MemoryStore, working: WorkingMemory,
             llm_complete=complete, embed_fn=None,
             retrieval_on: bool = True) -> TurnResult:
    """Execute one turn. Never raises: failures land in the trace as failed."""
    from embeddings import embed_query

    trace: list[TraceStep] = []
    t0 = time.perf_counter()

    def step(step_id: str, label: str, status: str, detail: str = "") -> None:
        trace.append(TraceStep(id=step_id, label=label, status=status,
                               ms=(time.perf_counter() - t0) * 1000, detail=detail))

    masked = mask(user_text)
    step("screen_input", "PII/secret screen on input", "ok",
         f"{sum(masked.counts.values())} value(s) masked" if masked.counts else "clean")
    try:
        context = [mask(turn).text for turn in working.read(session_id, limit=10)]
    except (OSError, ValueError) as exc:
        return TurnResult(answer="Working memory is unavailable; try again.",
                          persisted=False, working_saved=False, trace=trace + [
                              TraceStep(id="context", label="Read bounded Redis context",
                                        status="failed", detail=str(exc))])
    step("context", "Read bounded working context", "ok", f"{len(context)} turn(s)")

    try:
        query_vector = embed_fn(masked.text, settings) if embed_fn else embed_query(masked.text, settings)
    except (OSError, ValueError) as exc:
        step("embed", "Embed protected query locally", "failed", str(exc))
        return TurnResult(answer="Embedding is unavailable; try again.",
                          persisted=False, working_saved=False, trace=trace)
    step("embed", "Embed protected query locally", "ok")

    memories: list[StoredMemory] = []
    if retrieval_on:
        try:
            memories, _ = _screen_memories(
                store.search(user_id, query_vector, settings.top_k), trace)
            step("retrieve", "Top-k user-scoped retrieval", "ok", f"{len(memories)} kept")
        except (OSError, ValueError) as exc:
            step("retrieve", "Top-k user-scoped retrieval", "failed", str(exc))
    else:
        step("retrieve", "Retrieval disabled; memories excluded", "skipped")

    messages = build_prompt(masked.text, context, memories, retrieval_on)
    step("prompt", "Build bounded redacted prompt", "ok")
    try:
        result: ModelResult = llm_complete(messages, settings)
    except LLMError as exc:
        step("model_call", "One hosted call", "failed", str(exc)[:200])
        return TurnResult(answer="The model response was unusable; nothing was stored.",
                          persisted=False, working_saved=False, trace=trace)
    step("model_call", "One hosted call", "ok")

    # Answer is user-facing: restore nothing (mapping holds no originals for
    # input the user typed), but screen it before use.
    if contains_pii(result.answer):
        step("validate", "Answer contained PII; withheld", "failed",
             "withheld answer with detected values")
        return TurnResult(answer="I cannot show that answer as-is; try rephrasing.",
                          persisted=False, working_saved=False, trace=trace)

    decisions: list[MemoryEvent] = []
    persisted_any = False
    for proposal in result.proposals:
        decisions.append(_apply_proposal(proposal, user_id, query_vector,
                                         store, settings, trace, masked))
        persisted_any = persisted_any or decisions[-1].type in ("ADD", "SUPERSEDE")
    step("validate", "Parse, screen, apply policy", "ok",
         f"{len([d for d in decisions if d.type != 'REJECT'])} kept, "
         f"{len([d for d in decisions if d.type == 'REJECT'])} rejected")
    try:
        for event in decisions:
            store.record_event(user_id, event)
        step("events", "Append memory events", "ok", f"{len(decisions)} event(s)")
    except (OSError, ValueError) as exc:
        step("events", "Append memory events", "failed", str(exc))

    try:
        working.append(session_id, f"user: {masked.text}")
        working.append(session_id, f"assistant: {mask(result.answer).text}")
        working_saved = True
        step("working", "Append turns per TTL", "ok")
    except (OSError, ValueError) as exc:
        working_saved = False
        step("working", "Append turns per TTL", "failed", str(exc))

    step("trace", "Return answer and complete trace", "ok")
    return TurnResult(answer=result.answer, persisted=persisted_any,
                      working_saved=working_saved, memories=memories,
                      decisions=[f"{d.type}: {d.reason}" for d in decisions],
                      events=decisions, trace=trace)


def _apply_proposal(proposal: Proposal, user_id: str, query_vector: list[float],
                    store: MemoryStore, settings: Settings,
                    trace: list[TraceStep], masked: MaskResult) -> MemoryEvent:
    from embeddings import embed as embed_texts

    if contains_pii(proposal.text):
        return MemoryEvent(type="REJECT", reason="proposal contained PII/secret")
    candidates = store.search(user_id, query_vector, settings.top_k)
    for cand in candidates:
        if cand.text.strip() == proposal.text.strip():
            return MemoryEvent(type="REJECT", reason="duplicate of stored memory")
        if cand.category == proposal.category and cand.score >= SUPERSEDE_SIMILARITY:
            vector = embed_texts([proposal.text], settings)[0]
            new_mem = store.add(user_id, proposal.text, proposal.category, vector)
            store.link_supersede(cand.id, new_mem.id)
            return MemoryEvent(type="SUPERSEDE", reason="contradicts stored memory",
                               memory_id=new_mem.id, supersedes=cand.id)
    vector = embed_texts([proposal.text], settings)[0]
    new_mem = store.add(user_id, proposal.text, proposal.category, vector)
    return MemoryEvent(type="ADD", reason="accepted by policy", memory_id=new_mem.id)
