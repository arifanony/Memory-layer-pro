# Memory Layer Pro

Teaching MVP: a GPT-style memory application showing how chat systems keep
working context, retrieve durable memories, and let Python decide what
persists. **Synthetic data only. Not production-ready.**

## Quickstart (local Mac, UV)

```bash
uv python install 3.13
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e ".[dev]"   # full stack (later tasks); scaffold needs pytest only
cp .env.example .env          # then fill NVIDIA_API_KEY and NVIDIA_MODEL
python -m pytest tests/ -q
```

PostgreSQL with pgvector and Redis run natively for build/test; Docker
packaging comes at the end (see ADR.md).

## Layout

- `app.py`: Streamlit entry point.
- `orchestrator.py`: turn pipeline and trace.
- `schemas.py`: validated response and memory contracts.
- `config.py`: environment loading and startup validation.
- `evaluation.py`: deterministic synthetic eval set (FR-12); run with `python evaluation.py`.
- `tests/`: focused checks, mocks only, no live key required.

## Docs

- [ADR.md](ADR.md): the brief this repo implements.
- [SPEC.md](SPEC.md): requirements (FR/NFR/AC) and acceptance criteria.
- [PIPELINE.md](PIPELINE.md): visual end-to-end Mermaid pipeline.
