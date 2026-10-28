"""Turn pipeline, policy, events, failure honesty (ORCHESTRATOR; FR-7/8/13, AC-2/5/12/13)."""

import pytest

from config import validate, load
from llm_client import LLMError, ModelResult, Proposal
from orchestrator import StoredMemory, build_prompt, run_turn


def _settings(**overrides):
    env = {"NVIDIA_API_KEY": "k", "NVIDIA_MODEL": "m"}
    env.update(overrides)
    return validate(load(env))


class FakeStore:
    def __init__(self, memories=(), fail_on=()):
        self.memories = {m.id: m for m in memories}
        self.links = []
        self.events = []
        self.fail_on = set(fail_on)
        self.next_id = 100

    def record_event(self, user_id, event):
        if "events" in self.fail_on:
            raise OSError("events down")
        self.events.append(event)

    def search(self, user_id, vector, top_k):
        if "search" in self.fail_on:
            raise OSError("pgvector down")
        assert user_id == "u1"
        return sorted(self.memories.values(), key=lambda m: -m.score)[:top_k]

    def add(self, user_id, text, category, vector):
        if "add" in self.fail_on:
            raise OSError("pgvector down")
        mem = StoredMemory(id=f"m{self.next_id}", text=text, category=category)
        self.next_id += 1
        self.memories[mem.id] = mem
        return mem

    def link_supersede(self, old_id, new_id):
        self.links.append((old_id, new_id))


class FakeWorking:
    def __init__(self, fail=False):
        self.turns = []
        self.fail = fail

    def read(self, user_id, session_id, limit):
        return self.turns[-limit:]

    def append(self, user_id, session_id, turn):
        if self.fail:
            raise OSError("redis down")
        self.turns.append(turn)

    def clear(self, user_id, session_id):
        self.turns.clear()

    def clear_user(self, user_id):
        self.turns.clear()
        return 0


def _llm(answer="done", proposals=()):
    def fake(messages, settings, timeout=60):
        fake.calls += 1
        fake.last_messages = messages
        return ModelResult(answer=answer,
                           proposals=tuple(Proposal(text=t, category=c) for t, c in proposals))
    fake.calls = 0
    return fake


def _embed(text, settings):
    return [0.1] * 384


def test_happy_path_persists_and_traces():
    llm = _llm("here", [("likes tea", "preference")])
    out = run_turn("remember I like tea", "u1", "s1", _settings(),
                   FakeStore(), FakeWorking(), llm_complete=llm, embed_fn=_embed)
    assert out.answer == "here" and out.persisted and out.working_saved
    assert llm.calls == 1
    assert any(e.type == "ADD" for e in out.events)
    assert all(s.status in ("ok", "skipped") for s in out.trace)


def test_duplicate_rejected_not_stored():
    store = FakeStore([StoredMemory(id="m1", text="likes tea", category="preference", score=0.5)])
    llm = _llm("ok", [("likes tea", "preference")])
    out = run_turn("remember", "u1", "s1", _settings(), store, FakeWorking(),
                   llm_complete=llm, embed_fn=_embed)
    assert not out.persisted
    assert out.events[0].type == "REJECT" and "duplicate" in out.events[0].reason
    assert len(store.memories) == 1


def test_contradiction_supersedes_with_link():
    store = FakeStore([StoredMemory(id="m1", text="lives in Berlin", category="fact", score=0.95)])
    llm = _llm("ok", [("lives in Paris", "fact")])
    out = run_turn("moved", "u1", "s1", _settings(), store, FakeWorking(),
                   llm_complete=llm, embed_fn=_embed)
    assert out.persisted
    assert out.events[0].type == "SUPERSEDE" and out.events[0].supersedes == "m1"
    assert store.links == [("m1", out.events[0].memory_id)]


def test_pii_proposal_rejected_never_stored():
    llm = _llm("ok", [("mail me at a@b.co", "fact")])
    store = FakeStore()
    out = run_turn("hi", "u1", "s1", _settings(), store, FakeWorking(),
                   llm_complete=llm, embed_fn=_embed)
    assert not out.persisted and len(store.memories) == 0
    assert out.events[0].type == "REJECT"


def test_input_masked_before_llm_and_trace_is_clean():
    llm = _llm("noted")
    out = run_turn("my mail is a@b.co", "u1", "s1", _settings(),
                   FakeStore(), FakeWorking(), llm_complete=llm, embed_fn=_embed)
    sent = llm.last_messages[1]["content"]
    assert "a@b.co" not in sent and "[EMAIL_1]" in sent
    assert "a@b.co" not in repr(out.trace)


def test_retrieval_off_excludes_memories():
    llm = _llm("ok")
    run_turn("hi", "u1", "s1", _settings(),
             FakeStore([StoredMemory(id="m1", text="secret-base", category="fact", score=1.0)]),
             FakeWorking(), llm_complete=llm, embed_fn=_embed, retrieval_on=False)
    assert "secret-base" not in llm.last_messages[1]["content"]


def test_tainted_memory_excluded_from_prompt():
    llm = _llm("ok")
    out = run_turn("hi", "u1", "s1", _settings(),
                   FakeStore([StoredMemory(id="m1", text="call 10.0.0.8", category="fact", score=1.0)]),
                   FakeWorking(), llm_complete=llm, embed_fn=_embed)
    assert "10.0.0.8" not in llm.last_messages[1]["content"]
    assert any(s.id == "screen_memory" for s in out.trace)


def test_store_failure_still_answers_but_flags_unsaved():
    llm = _llm("here")
    out = run_turn("hi", "u1", "s1", _settings(),
                   FakeStore(fail_on=("search",)), FakeWorking(),
                   llm_complete=llm, embed_fn=_embed)
    assert out.answer == "here" and not out.persisted
    assert any(s.status == "failed" for s in out.trace)


def test_bad_model_output_never_stores():
    def bad_llm(messages, settings, timeout=60):
        raise LLMError("not JSON")

    out = run_turn("hi", "u1", "s1", _settings(), FakeStore(), FakeWorking(),
                   llm_complete=bad_llm, embed_fn=_embed)
    assert not out.persisted and "nothing was stored" in out.answer


def test_working_failure_flagged():
    llm = _llm("here", [("likes tea", "preference")])
    out = run_turn("remember", "u1", "s1", _settings(), FakeStore(), FakeWorking(fail=True),
                   llm_complete=llm, embed_fn=_embed)
    assert out.persisted and not out.working_saved
    assert any(s.id == "working" and s.status == "failed" for s in out.trace)


def test_proposal_similarity_uses_proposal_embedding_not_query():
    seen = []

    class VecStore(FakeStore):
        def search(self, user_id, vector, top_k):
            seen.append(list(vector))
            return [StoredMemory(id="m1", text="lives in Berlin",
                                 category="fact", score=0.95)]

    def text_embed(text, settings):
        return [1.0 if "paris" in text.lower() else 0.0] * 384

    llm = _llm("ok", [("lives in Paris", "fact")])
    out = run_turn("moved house", "u1", "s1", _settings(), VecStore(),
                   FakeWorking(), llm_complete=llm, embed_fn=text_embed)
    # First search runs the query vector; the policy search must run the
    # proposal's own vector, which is what triggers the SUPERSEDE here.
    assert [1.0] * 384 in seen
    assert out.events[0].type == "SUPERSEDE"


def test_add_failure_still_answers_and_traces_exactly():
    llm = _llm("here", [("likes tea", "preference")])
    out = run_turn("hi", "u1", "s1", _settings(), FakeStore(fail_on=("add",)),
                   FakeWorking(), llm_complete=llm, embed_fn=_embed)
    assert out.answer == "here" and not out.persisted
    assert any(s.id == "persist" and s.status == "failed" for s in out.trace)


def test_events_failure_after_add_reports_what_succeeded():
    llm = _llm("here", [("likes tea", "preference")])
    out = run_turn("hi", "u1", "s1", _settings(), FakeStore(fail_on=("events",)),
                   FakeWorking(), llm_complete=llm, embed_fn=_embed)
    assert out.answer == "here" and out.persisted and out.working_saved
    assert any(s.id == "events" and s.status == "failed" for s in out.trace)


def test_build_prompt_redacted_and_toggleable():
    msgs = build_prompt("[EMAIL_1]", ["user: hi"],
                        [StoredMemory(id="m1", text="likes tea", category="preference")], True)
    assert "likes tea" in msgs[1]["content"]
    msgs_off = build_prompt("hi", [], [StoredMemory(id="m1", text="likes tea", category="fact")], False)
    assert "likes tea" not in msgs_off[1]["content"]
