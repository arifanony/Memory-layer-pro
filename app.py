"""Memory Layer Pro — Streamlit chat and pipeline trace (UI).

Teaching demo only: synthetic data, no personal information. Every assistant
reply carries an expandable trace (steps, retrievals, decisions, events) so a
learner can see exactly what the pipeline did. No raw secrets or unmasked PII
ever reach the page: values are masked before storage, prompts, and display.
"""

from __future__ import annotations

import time

import streamlit as st

from config import ConfigError, Settings, load, validate
from memory_store import PgStore, connect as pg_connect, init_schema
from orchestrator import TurnResult, run_turn
from schemas import MemoryEvent, StoredMemory
from working_memory import RedisWorking, connect as redis_connect


# --- Pure presentation helpers (covered by tests/test_ui_trace.py). ---

def format_ms(value: float) -> str:
    return f"{value * 1000:.0f} ms" if value >= 1 else f"{value * 1000:.1f} ms"


def status_label(status: str) -> str:
    return {"ok": "done", "skipped": "skipped", "failed": "failed"}.get(status, status)


def memory_line(mem: StoredMemory) -> str:
    return f"[{mem.category}] {mem.text} — similarity {mem.score:.2f}"


def event_line(event: MemoryEvent) -> str:
    target = f" → supersedes {event.supersedes}" if event.supersedes else ""
    return f"{event.type}{target}: {event.reason}"


def failed_steps(trace) -> list:
    return [s for s in trace if s.status == "failed"]


# --- Backend wiring (cached per session). ---

@st.cache_resource(show_spinner="Connecting to project services…")
def _backend():
    settings = validate(load())
    pg_conn = pg_connect(settings)
    init_schema(pg_conn)
    redis_client = redis_connect(settings)
    redis_client.ping()
    return settings, PgStore(pg_conn), RedisWorking(redis_client, settings.redis_ttl_seconds)


def main() -> None:
    st.set_page_config(page_title="Memory Layer Pro", layout="wide")
    st.title("Memory Layer Pro")
    st.caption("Teaching demo — synthetic data only. Nothing here is production-ready.")

    try:
        settings, store, working = _backend()
    except Exception as exc:
        st.error(f"Cannot start: {exc}")
        st.info("Copy .env.example to .env, fill the keys, and start services "
                "with scripts/setup-services.sh. Then reload this page.")
        st.stop()
        return

    with st.sidebar:
        st.header("Session")
        user_id = st.text_input("Demo user", value="demo-user")
        session_id = st.text_input("Session", value="demo-session")
        retrieval_on = st.checkbox("Retrieval", value=True,
                                   help="Off proves durable memories never reach the prompt.")
        top_k = st.slider("Top-k", 1, 10, settings.top_k)
        settings = settings.__class__(**{**settings.__dict__, "top_k": top_k})
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Clear session"):
                working.clear(user_id, session_id)
                st.session_state.pop("history", None)
                st.rerun()
        with col_b:
            forget = st.button("Forget user")
        if forget:
            # Full user reset: durable memories and events plus every
            # working session for this user. Clear session stays scoped
            # to the active session only.
            try:
                counts = store.forget_user(user_id)
                working.clear_user(user_id)
            except OSError as exc:
                st.error(f"Forget failed safely: {exc}. Demo state was not fully cleared.")
            else:
                st.session_state.pop("history", None)
                st.success(f"Forgot {counts['memories']} memories, {counts['events']} events.")
                st.rerun()
        st.divider()
        st.header("Services")
        st.write("Postgres + pgvector: connected")
        st.write("Redis: connected")
        st.write(f"Model: `{settings.nvidia_model or 'not set'}`")

    history = st.session_state.setdefault("history", [])
    for turn in history:
        with st.chat_message(turn["role"]):
            st.write(turn["content"])
            if turn["role"] == "assistant" and turn.get("detail"):
                _render_detail(turn["detail"])

    prompt = st.chat_input("Ask or teach something (synthetic only)…")
    if not prompt:
        return
    history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"), st.spinner("Thinking (one hosted call)…"):
        started = time.perf_counter()
        try:
            result: TurnResult = run_turn(prompt, user_id, session_id, settings,
                                          store, working, retrieval_on=retrieval_on)
        except (OSError, ValueError) as exc:
            st.error(f"Turn failed safely: {exc}. Nothing was stored.")
            return
        detail = {
            "answer": result.answer,
            "memories": [memory_line(m) for m in result.memories],
            "decisions": list(result.decisions),
            "events": [event_line(e) for e in result.events],
            "steps": [(s.id, s.label, status_label(s.status), format_ms(s.ms), s.detail)
                      for s in result.trace],
            "persisted": result.persisted,
            "working_saved": result.working_saved,
            "elapsed": format_ms(time.perf_counter() - started),
        }
    problems = failed_steps(result.trace)
    if problems:
        st.warning("Something downstream failed — the answer is shown but "
                   + ("memory was not saved." if not result.persisted else "see trace."))
    st.write(result.answer)
    _render_detail(detail)
    history.append({"role": "assistant", "content": result.answer, "detail": detail})


def _render_detail(detail: dict) -> None:
    with st.expander("Why this answer — pipeline trace"):
        steps = detail["steps"]
        st.write(f"{len([s for s in steps if s[2] == 'done'])} of {len(steps)} steps done · "
                 f"persisted: {detail['persisted']} · working context saved: "
                 f"{detail['working_saved']} · turn took {detail['elapsed']}")
        st.table([{"step": label, "status": status, "time": when}
                  for _, label, status, when, _ in steps])
        if detail["memories"]:
            st.subheader("Retrieved memories")
            for line in detail["memories"]:
                st.write("- " + line)
        else:
            st.write("No durable memories used (retrieval off or none relevant).")
        if detail["decisions"]:
            st.subheader("Memory decisions and events")
            for line in detail["decisions"] + detail["events"]:
                st.write("- " + line)
        failed = [s for s in steps if s[2] == "failed"]
        if failed:
            with st.expander("Failure detail (redacted)"):
                for step_id, label, _, _, note in failed:
                    st.write(f"{step_id} ({label}): {note}")


if __name__ == "__main__":
    main()
