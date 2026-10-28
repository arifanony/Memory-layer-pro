# Memory Layer Diagrams — current system vs mem0 vs ChatGPT

Synthetic teaching content only. `[*]` = planned, not built yet.

## 1. Our model today (Memory Layer Pro)

```
USER TURN ("I like roast tea")
  │
  ▼
┌─────────────────────────┐
│ 1. screen_input         │  mask PII → [EMAIL_1] etc. (or "clean")
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ 2. context (Redis)      │  read last N turns, key wm:{session}
│    Clear session wipes  │  ONLY this box — Postgres untouched
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ 3. embed (local bge)    │  same model + preprocessing as stored rows
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ 4. retrieve (pgvector)  │  top-k, ALWAYS scoped to user_id
│    categories today:    │  fact · preference · procedure
│    [*] episodic [*]     │  semantic (rename of fact) — planned
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ 5. screen_memories      │  drop retrieved rows containing PII
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ 6. prompt               │  masked input + context + memories
│    retrieval OFF →      │  memories provably excluded
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ 7. ONE hosted call      │  answer + memory_proposals (JSON)
│    (NVIDIA Nemotron)    │  exactly 1 call per successful turn
└────────────┬────────────┘
             ▼
┌─────────────────────────────────────────┐
│ 8. DETERMINISTIC GATE (Python, no LLM)  │  ◄── the "judge" today
│   proposal → PII? → REJECT              │
│            → duplicate? → REJECT        │
│            → contradicts same-category? │
│                 → SUPERSEDE (link old→new)│
│            → else ADD                   │
│   [*] general-knowledge / ephemeral     │
│       gate — MISSING (why list/Counter  │
│       rows got stored)                  │
└────────────┬────────────────────────────┘
             ▼
┌─────────────────────────┐
│ 9. events + working     │  one event row per decision;
│                         │  turns appended to Redis (TTL)
└────────────┬────────────┘
             ▼
  ANSWER + TRACE (per-stage verdicts, scores, reasons)
```

## 2. How each block is processed (inputs → check → output)

```
screen_input : raw text → PII scan → masked text + counts
context      : session_id → Redis LRANGE → last N turns
embed        : masked text → 384-dim vector (local model)
retrieve     : vector + user_id + top_k → ranked memories + scores
               score >= 0.50 = strong · below = weak (label only)
screen_mem   : each memory → PII scan → keep or drop with reason
prompt       : masked + context + kept memories → messages list
model_call   : messages → ModelResult(answer, proposals)
validate     : answer PII scan → withhold or keep
               each proposal → gate table (ADD / SUPERSEDE / REJECT + reason)
events       : each decision → memory_events row (user-scoped)
working      : user turn + assistant turn → Redis list (bounded, TTL)
```

## 3. How mem0 does it

```
USER TURN
  │
  ▼
memory.add(messages)
  │
  ├─► EXTRACTION (LLM call 1) ──► candidate facts
  │
  ├─► search similar stored memories (vector)
  │
  ├─► UPDATE (LLM call 2, function-calling)
  │     per fact → ADD · UPDATE · DELETE · NOOP + reason
  │     (newer versions: single-pass additive, fewer rewrites)
  │
  ▼
vector store (Qdrant/Chroma/…) — facts as embeddings

memory.search(query) ──► semantic search ──► ranked facts
```

Judge = two LLM calls. Forgetting (UPDATE/DELETE/NOOP) is first-class —
that operation is what our gate is missing.

## 4. How ChatGPT does it

```
CONVERSATION
  │
  ▼
background EXTRACTOR (model-driven, mostly unprompted —
research: ~96% of memories saved without being asked)
  │
  ▼
┌──────────────────────┐   ┌────────────────────────┐
│ SAVED MEMORIES       │   │ CHAT HISTORY REFERENCE │
│ discrete facts list  │   │ broad past-chat signal │
│ viewable · editable  │   │ not a browsable list   │
│ deletable per item   │   │                        │
└──────────┬───────────┘   └───────────┬────────────┘
           └──────────────┬────────────┘
                          ▼
           prepended to every new chat as context
```

Judge = the model itself. Rectification = user visibility and control
(view/edit/delete), not a stricter verdict algorithm.

## 5. What remains in our build

- Deterministic general-knowledge/ephemeral gate (the gap)
- `fact` → `semantic` rename (+ episodic layer) with row migration
- Per-proposal verdict table in the trace UI
- FR-11: seed data per layer + labeled mock mode for keyless walkthroughs
