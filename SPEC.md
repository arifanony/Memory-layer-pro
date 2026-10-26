# Product specification — Memory Layer Pro

> Status: draft. A coding agent must not implement product code until this specification is approved through Genesis (`genesis spec approve`).
> Source: [ADR.md](/Users/arif/Projects/Memory layer/ADR.md) (Memory Lab MVP brief). Local teaching app, synthetic data only.

## Problem

Learners cannot see how a GPT-style memory system works end to end: what is kept in
working context, what is retrieved from durable memory, how one model call returns both
an answer and memory proposals, and how Python (not another model call) decides what
persists. Memory Layer Pro demonstrates that pipeline with a readable per-stage trace.

## Users

- Learner operating the Streamlit UI with a synthetic demo user/session.
- No real end users, no personal data. Every fact stored is synthetic and deletable.

## Functional requirements

- FR-1: Select or create a synthetic demo user/session; provide clear-session and forget/reset controls for demo state.
- FR-2: Screen the incoming message locally for PII/secrets and mask values with typed placeholders (`[EMAIL_1]`) before logs, traces, storage, or hosted prompts.
- FR-3: Read a bounded sequence of recent turns from Redis under a session/user-scoped key with configurable TTL; apply FR-2 screening to any context entering a prompt.
- FR-4: Embed the protected query with the configured local embedding model and preprocessing, identical to the memory path.
- FR-5: Retrieve top-k durable memories from pgvector, always scoped to the active demo user; show similarity/distance and safe metadata.
- FR-6: Make exactly one hosted Nemotron chat-completions call per successful turn, requesting a structured object with a required answer plus zero or more memory proposals.
- FR-7: Parse and validate the model response in Python; screen answer and proposals; apply deterministic policy (reject empty, duplicate, disallowed-category, overlong, PII/secret-flagged, with reasons); embed and persist accepted proposals.
- FR-8: Append protected turns to Redis per TTL; never report a write as successful unless the database confirms it; on persistence failure still show the answer but mark memory unsaved.
- FR-9: Streamlit UI shows chat, user/session controls, retrieval on/off plus top-k, retrieved memories, proposal decisions with reasons, per-step trace with durations, redacted technical detail, error states, and clear/reset controls.
- FR-10: Retrieval-disabled path provably excludes durable memories from the model prompt.
- FR-11: Ship synthetic seed data and a labeled mock mode for UI walkthroughs when hosted inference is unavailable; never present mock output as live inference.
- FR-12: Provide a deterministic synthetic evaluation set and report hit@k, accepted/rejected counts, invalid outputs, and per-stage latency.

## Non-functional requirements

- NFR-1: One model call per successful turn; no second LLM extraction call. A single repair retry only if it fits the teaching objective; default is to report invalid output.
- NFR-2: Same local embedding model and preprocessing on query and memory paths; actionable error when unloadable; never silently fall back to hosted embeddings.
- NFR-3: Parameterized SQL; every read, write, and delete scoped by user id; pgvector enabled via documented setup.
- NFR-4: No original PII/secret values in logs, traces, UI, or storage; placeholder mapping stays request-scoped, default no restore.
- NFR-5: Automated checks pass with mocks and no live NVIDIA key; live-service smoke is a separate manual step.
- NFR-6: Startup validates configuration and fails with actionable errors for missing provider, Redis, PostgreSQL, embedding, or parsing setups.

## Constraints

- Synthetic data only; pattern-based PII detection is best-effort and the app must not claim production privacy or compliance.
- NVIDIA API catalog models, eligibility, and rate limits change; endpoint and model stay environment-configured.
- Requires local PostgreSQL with pgvector, Redis, and a downloadable embedding model (Docker path documented).
- Secrets stay in environment, never in git.

## Non-goals

- Production auth, multi-tenant accounts, or authorization system.
- Production hosting, HA, autoscaling, backup/DR, or ops dashboards.
- A second LLM call for extraction or evaluation.
- Autonomous model-directed deletion or unrestricted updates.
- Knowledge graphs, entity resolution, reranking, consolidation jobs, advanced temporal reasoning.
- Claims of free inference, unlimited access, fixed quotas, or production-grade PII safety.

## Acceptance criteria

- AC-1: App starts with documented config and reports missing services clearly.
- AC-2: A turn retrieves scoped durable memories plus bounded Redis context.
- AC-3: One successful turn uses one hosted call for answer and proposals.
- AC-4: Query and memory embeddings use the same local model path.
- AC-5: Python validates the response and decides persistence.
- AC-6: PII/secret-like values are masked pre-request and rejected from storage.
- AC-7: UI exposes pipeline, retrieval, decisions, durations, failures without raw sensitive values.
- AC-8: Retrieval-off mode demonstrably excludes durable memories from the prompt.
- AC-9: Clear/forget controls work for the active demo session/user.
- AC-10: Automated checks pass without a real NVIDIA key.
- AC-11: README matches implementation, limits, hosted-access caveats, synthetic-data rule.

## Risks

- R-1: Selected Nemotron model unavailable or rate-limited; mitigation: configurable model/endpoint plus clear provider error.
- R-2: PII detector misses or false-positives; mitigation: synthetic-only rule, reject-on-flag default.
- R-3: Embedding dimension drift vs pgvector column; mitigation: single configured model, migration docs.

## Open questions

- Q-1: NVIDIA API key: the operator earlier required keyless operation, but this spec needs a hosted key for live turns. Decision needed: obtain key or accept mock-only teaching mode.
- Q-2: Are Docker PostgreSQL/pgvector plus Redis approved to run on this Mac?
- Q-3: Which Nemotron model id should be the default in `.env.example`?
