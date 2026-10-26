# Memory Lab MVP — Implementation Brief

## Instruction to the implementation agent

Build the project described below in the current repository. First inspect the existing files and follow the repository's conventions. If this is an empty repository, create a small, readable Python application from scratch.

Implement the MVP end to end. Make reasonable engineering choices where details are open, document those choices, and keep the system easy to teach and inspect. Do not stop after producing another plan. Do not add production-scale infrastructure or unrelated features. Keep secrets out of source control. Run the checks needed to confirm the implementation works and report what was verified and any external services that still need user configuration.

## Product goal

Create an educational GPT-style memory application that demonstrates how a chat system:

- keeps recent working context,
- retrieves durable memories relevant to a new question,
- uses those memories to answer,
- proposes new memories alongside the answer in the same model request,
- validates and filters memory proposals in Python,
- exposes each stage in a clear UI trace.

The app is for local teaching and experimentation. It must use synthetic example data by default and clearly identify itself as an MVP, not a production-ready personal-data service.

## Required architecture

- **Python application and orchestration.**
- **Streamlit UI** for chat, controls, and a readable, step-by-step pipeline trace.
- **NVIDIA hosted Nemotron API** through its OpenAI-compatible chat-completions interface. Configure the endpoint, model, and key through environment variables. Do not hard-code a model identifier that may become unavailable.
- **One model call per successful user turn** to generate both the user-facing answer and structured memory proposals. Python performs validation and policy decisions; it does not make a second LLM extraction/evaluator call.
- **PostgreSQL with pgvector** for durable, user-scoped memory records and vector retrieval.
- **Redis** for short-lived working conversation context with a configurable TTL and bounded history.
- **Local embedding model** for both memory and query embeddings. Use the same configured model and preprocessing on both paths.
- **Python-side validation and evaluation** for model output, memory policy, and a small synthetic evaluation set.

Keep modules and abstractions minimal. A reasonable starting layout is:

- `app.py`: Streamlit entry point and presentation.
- `orchestrator.py`: turn pipeline and trace events.
- `llm_client.py`: hosted API client and structured response handling.
- `memory_store.py`: PostgreSQL/pgvector access.
- `working_memory.py`: Redis conversation state.
- `embeddings.py`: local embedding model.
- `privacy.py`: PII/secret detection and masking.
- `schemas.py`: validated request/response and memory types.
- `evaluation.py`: small repeatable evaluation.
- `tests/`: focused unit and integration checks.
- `.env.example`, `.gitignore`, and `README.md`: configuration and usage.

Adapt the layout to existing repository structure when appropriate.

## Turn pipeline

Implement the following sequence and expose its progress to the UI:

1. Select or create a synthetic demo user/session.
2. Inspect the incoming message locally for PII and secret-like values. Mask detected values before logs, traces, storage, or hosted prompts.
3. Read bounded recent turns from Redis and apply the same privacy screening to any content that might enter a prompt.
4. Embed the protected query locally.
5. Retrieve a small top-k set of durable memories from pgvector, always scoped to the active demo user.
6. Screen retrieved memories and construct a bounded prompt from the protected current message, recent context, and selected memories.
7. Make one Nemotron API call. Request a structured object containing an answer and zero or more memory proposals.
8. Parse and validate the response in Python. Screen the answer and proposals for PII/secrets.
9. Apply deterministic memory policy. Reject unsafe, empty, duplicate, disallowed, or overlong proposals and record clear reasons.
10. Embed and write accepted proposals to PostgreSQL.
11. Append the protected user turn and assistant answer to Redis according to the storage policy and TTL.
12. Return the answer and complete trace. Never report a write as successful unless the database confirms it.

If a stage fails, preserve the trace up to the failure, show an understandable error, and do not claim later stages succeeded. A generated answer may still be displayed if only memory persistence fails, but the UI must explicitly show that memory was not saved.

## Memory proposal contract

Define and enforce a small JSON schema for the model result. It should include:

- `answer`: required non-empty string with a configured maximum length.
- `memory_proposals`: array with a configured maximum number of items.
- Each proposal: concise `text`, allowlisted `category`, and optional bounded confidence or operation fields.

Keep the schema small and explain it in the README. Prefer constrained structured output when the chosen model endpoint supports it. Otherwise request JSON and parse strictly. Do not persist malformed output. A single repair retry may be used only if it fits the one-call-per-turn teaching objective; default behavior should be to report invalid output without silently making another LLM call.

## Memory storage behavior

### Durable memory: PostgreSQL + pgvector

Store at minimum:

- memory id,
- demo user id,
- text,
- category,
- embedding vector,
- created and updated timestamps,
- source turn id,
- optional expiry timestamp.

Use parameterized SQL. Scope every read, update, and delete by user id. Enable the vector extension through documented setup/migration steps. Select an appropriate vector distance operator and index configuration for the chosen embedding model. For a tiny dataset, exact vector search is acceptable; document when an approximate index is useful.

### Working memory: Redis

Store a bounded sequence of recent turns under a session/user-scoped key. Apply a configurable TTL. Do not treat Redis as durable memory. Provide a clear-session operation.

### Memory policy

For this MVP, Python controls persistence. Reject proposals that are empty, excessively long, duplicate, outside allowed categories, ephemeral when the policy excludes them, or contain detected PII/secrets. Make accepted and rejected decisions visible in the trace. Provide a forget/reset operation for demo data.

## PII checks and security masking

Implement a local privacy guard that runs before any hosted request. Keep the detector behind a small interface so the implementation can be strengthened later.

The baseline detector should recognize common patterns for:

- email addresses,
- phone numbers,
- IP addresses,
- payment-card-like numbers,
- government-ID-like patterns appropriate to the demo locale,
- credentials, API keys, and obvious secret tokens.

Requirements:

- Replace detected values with typed placeholders such as `[EMAIL_1]` and `[PHONE_1]` before hosted prompts, trace payloads, and ordinary logs.
- Apply screening to user input, recent conversation context, retrieved memories, model answers, and proposed memories.
- Reject flagged memory proposals by default. Never save detected PII/secret values to PostgreSQL or Redis.
- Do not display original sensitive values in the UI trace. Show the detected category, count, and masking/rejection result.
- Avoid raw prompt and response logging. Never log API keys or connection credentials.
- If response restoration is implemented, keep the placeholder mapping only in request-scoped memory, restore only in the user-facing response, and discard the mapping after the turn. The safer default is not to restore.
- Document that pattern matching can miss PII and can produce false positives. The app must use synthetic data and must not claim compliance or protection suitable for real personal data.

Do not send a detected original value to the hosted endpoint for debugging or evaluation.

## Streamlit UI requirements

Build a usable single-page application with:

- chat history and message input,
- a synthetic demo user/session selector or clear session action,
- controls for retrieval on/off and top-k if appropriate,
- visible retrieved memories with similarity/distance and safe metadata,
- visible memory proposal decisions, including rejection reasons,
- a pipeline trace with per-step status, elapsed duration, and short explanation,
- expandable technical detail for safe, redacted payloads,
- clear error states for provider, Redis, PostgreSQL, embedding, parsing, and validation failures,
- clear/reset controls for working context and demo durable memories.

Keep the normal view easy for a learner to follow. Do not show secrets or unmasked PII in trace details. Include a retrieval-disabled comparison path that guarantees durable memories are not added to the model prompt.

## Configuration and setup

Provide a `.env.example` with placeholders only:

```dotenv
NVIDIA_API_KEY=
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL=
DATABASE_URL=postgresql://user:password@localhost:5432/memory_lab
REDIS_URL=redis://localhost:6379/0
EMBEDDING_MODEL=
TOP_K=3
REDIS_TTL_SECONDS=3600
```

Add appropriate configuration validation and actionable startup errors. Never commit real credentials. Add environment files, local database files, model caches, and generated private data to `.gitignore`.

Document:

- supported Python version and install command,
- how to configure NVIDIA API Catalog access and select a currently available Nemotron model,
- how to start PostgreSQL with pgvector and Redis,
- how to initialize the database,
- how to run the app and checks,
- how to load or enter synthetic seed memories,
- how to clear demo state,
- known limitations and fallback operation.

NVIDIA's API Catalog currently documents an OpenAI-compatible hosted API pattern, including `https://integrate.api.nvidia.com/v1`. Endpoint availability, model choices, eligibility, terms, and rate limits can change. Treat hosted free/trial access as limited and variable, never unlimited and never as a guaranteed quota. Keep model and endpoint configurable and document that the user must check current NVIDIA catalog/account limits before use.

## Evaluation and checks

Create a small, deterministic synthetic evaluation set that covers:

- a fact or preference that should be retrieved,
- an unrelated question that should not retrieve it as useful context,
- an ephemeral statement that should not become durable memory,
- a proposal rejected by policy,
- retrieval enabled versus disabled.

Report simple measures such as retrieval hit@k, proposal accepted/rejected counts, invalid output count, and per-stage latency. Keep evaluation data synthetic and do not include raw PII.

Implement focused tests/checks for:

- schema validation, size limits, category allowlist, duplicate handling, and policy reasons,
- PII/secret detection and masking, including representative synthetic examples and false positives,
- verification that masking occurs before an outbound request,
- verification that raw detected values do not appear in logs/traces or durable/working storage,
- Redis ordering, bounded history, TTL, and clear behavior,
- PostgreSQL insert, user-scoped retrieval, and deletion,
- retrieval-disabled prompt construction,
- malformed model output and provider/storage failure handling,
- UI trace status and secret redaction where practical.

Use mocks for provider calls in automated tests. Do not require a live API key for the normal test suite. Clearly distinguish automated checks from a manual smoke test that needs configured services.

## Acceptance criteria

The implementation is complete when:

- The app starts with documented configuration and reports missing services or settings clearly.
- A user turn can retrieve scoped durable memories and bounded Redis context.
- One successful turn uses one hosted Nemotron chat-completions request for both answer and memory proposals.
- Local embeddings are used consistently for query and memory vectors.
- Python validates the model response and decides whether proposed memories may be persisted.
- PII/secret-like values are masked before hosted requests and rejected from storage by default.
- The UI exposes the pipeline, retrieval results, proposal decisions, durations, and failures without revealing raw sensitive values.
- Retrieval can be turned off and the prompt builder demonstrably excludes durable memories in that mode.
- Clear/forget controls work for the active demo session/user.
- Automated checks pass without requiring a real NVIDIA API key.
- README setup steps, limitations, hosted-access caveats, and synthetic-data guidance match the implementation.

## Explicit non-goals

Do not add these unless the repository already requires them or the user requests them:

- production authentication, multi-tenant account management, or authorization system,
- production hosting, high availability, autoscaling, backup/disaster recovery, or operations dashboards,
- a second LLM call for extraction or evaluation,
- autonomous model-directed deletion or unrestricted updates,
- knowledge graphs, sophisticated entity resolution, reranking, consolidation jobs, or advanced temporal reasoning,
- claims of guaranteed free inference, unlimited access, or a fixed quota,
- claims of production privacy, regulatory compliance, or safe handling of real personal data.

## Fallback behavior

- If the selected NVIDIA model is unavailable or rate-limited, allow configuration of another currently available compatible model/provider and show the provider error clearly.
- If hosted inference cannot be configured, provide a mock mode only for UI walkthroughs and label every mock response clearly; never represent it as live inference.
- If Redis is unavailable, fail clearly by default. An explicitly enabled single-process session-state fallback may be provided for teaching, and must be labeled as non-Redis behavior.
- If pgvector initialization fails, show the setup error. A temporary in-memory retrieval demo is optional only when visibly labeled and must not count as completion of the required architecture.
- If the local embedding model cannot load, show an actionable error; do not silently switch to a hosted embedding service.
- If privacy screening detects PII, redact it before external calls and reject storage by default. If detector accuracy is uncertain, retain the synthetic-data-only rule.

## Handoff report

After implementation, report:

- the main files created or changed,
- how to configure and run the app,
- the checks run and their results,
- any manual setup still required (API key, database, Redis, model download),
- known limitations, especially PII detector coverage and hosted endpoint limits.

