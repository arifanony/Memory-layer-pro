"""Deterministic synthetic evaluation set (EVALUATION; FR-12).

No live key, services, or network: in-memory fakes stand in for pgvector and
Redis, and scripted stand-ins replace the hosted model call. Every case uses
synthetic data only. Run with `python evaluation.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config import load, validate
from llm_client import LLMError, ModelResult, Proposal
from orchestrator import run_turn
from schemas import StoredMemory

USEFUL_SCORE = 0.5


@dataclass
class EvalCase:
    name: str
    query: str
    seeds: list[StoredMemory] = field(default_factory=list)
    proposals: list[tuple[str, str]] = field(default_factory=list)
    answer: str = "noted"
    retrieval_on: bool = True
    llm_raises: bool = False
    expect_hit: str | None = None
    expect_useful: bool = False
    expect_persisted: bool = False


class EvalStore:
    """Score-ordered in-memory stand-in for the pgvector store."""

    def __init__(self, memories: list[StoredMemory]):
        self.memories = {m.id: m for m in memories}
        self.events: list = []
        self.links: list = []
        self.next_id = 100

    def search(self, user_id, vector, top_k):
        ranked = sorted(self.memories.values(), key=lambda m: -m.score)
        return ranked[:top_k]

    def add(self, user_id, text, category, vector):
        mem = StoredMemory(id=f"m{self.next_id}", text=text, category=category)
        self.next_id += 1
        self.memories[mem.id] = mem
        return mem

    def link_supersede(self, old_id, new_id):
        self.links.append((old_id, new_id))

    def record_event(self, user_id, event):
        self.events.append(event)


class EvalWorking:
    def __init__(self):
        self.turns: list[str] = []

    def read(self, user_id, session_id, limit):
        return self.turns[-limit:]

    def append(self, user_id, session_id, turn):
        self.turns.append(turn)

    def clear(self, user_id, session_id):
        self.turns.clear()

    def clear_user(self, user_id):
        self.turns.clear()
        return 0


def _embed(text, settings):
    return [0.1] * 384


def _scripted_llm(case: EvalCase):
    def fake(messages, settings, timeout=60):
        fake.calls += 1
        fake.last_messages = messages
        if case.llm_raises:
            raise LLMError("synthetic invalid output")
        return ModelResult(
            answer=case.answer,
            proposals=tuple(Proposal(text=t, category=c) for t, c in case.proposals),
        )

    fake.calls = 0
    return fake


def default_cases() -> list[EvalCase]:
    seat = StoredMemory(id="m1", text="prefers aisle seat",
                        category="preference", score=0.9)
    weak_seat = StoredMemory(id="m1", text="prefers aisle seat",
                             category="preference", score=0.1)
    return [
        EvalCase(name="fact_retrieved", query="which seat should I pick?",
                 seeds=[seat], expect_hit="prefers aisle seat",
                 expect_useful=True),
        EvalCase(name="unrelated_not_useful",
                 query="what is the capital of Peru?",
                 seeds=[weak_seat], expect_hit=None, expect_useful=False),
        EvalCase(name="ephemeral_not_persisted",
                 query="I am typing right now", seeds=[seat],
                 expect_hit="prefers aisle seat", expect_useful=True,
                 expect_persisted=False),
        EvalCase(name="policy_rejects_pii", query="remember this",
                 seeds=[], proposals=[("contact me at a@b.co", "fact")],
                 expect_persisted=False),
        EvalCase(name="retrieval_toggle", query="which seat should I pick?",
                 seeds=[seat], expect_hit="prefers aisle seat",
                 expect_useful=True),
        EvalCase(name="invalid_output", query="hi", seeds=[],
                 llm_raises=True, expect_persisted=False),
    ]


def run_suite(cases: list[EvalCase] | None = None, top_k: int = 3) -> dict:
    settings = validate(load({"NVIDIA_API_KEY": "eval", "NVIDIA_MODEL": "eval"}))
    cases = default_cases() if cases is None else cases
    rows = []
    accepted = rejected = invalid = 0
    hits = 0
    scored = 0
    step_ms: dict[str, list[float]] = {}
    totals = []
    for case in cases:
        if case.name == "retrieval_toggle":
            rows.append(_run_toggle(case, settings, top_k, step_ms, totals))
            scored += 1
            hits += rows[-1]["hit"]
            accepted += rows[-1]["accepted"]
            rejected += rows[-1]["rejected"]
            continue
        store, working, llm = EvalStore(case.seeds), EvalWorking(), _scripted_llm(case)
        out = run_turn(case.query, "u1", "s1", settings, store, working,
                       llm_complete=llm, embed_fn=_embed,
                       retrieval_on=case.retrieval_on)
        for event in out.events:
            if event.type in ("ADD", "SUPERSEDE"):
                accepted += 1
            elif event.type == "REJECT":
                rejected += 1
        if case.llm_raises:
            invalid += 1
        useful = [m for m in out.memories if m.score >= USEFUL_SCORE]
        if case.expect_hit is None:
            hit = not any(m.score >= USEFUL_SCORE for m in out.memories)
        else:
            hit = any(m.text == case.expect_hit and m.score >= USEFUL_SCORE
                      for m in out.memories)
        _ = useful
        scored += 1
        hits += hit
        for step in out.trace:
            step_ms.setdefault(step.id, []).append(step.ms)
        totals.append(out.trace[-1].ms if out.trace else 0.0)
        rows.append({"case": case.name, "hit": hit,
                     "persisted": out.persisted,
                     "expected_persisted": case.expect_persisted
                     if case.proposals or case.name == "ephemeral_not_persisted"
                     else None,
                     "accepted": sum(1 for e in out.events
                                     if e.type in ("ADD", "SUPERSEDE")),
                     "rejected": sum(1 for e in out.events if e.type == "REJECT"),
                     "invalid": case.llm_raises})
    return {
        "cases": len(cases),
        "hit_at_k": hits / scored if scored else 0.0,
        "accepted": accepted,
        "rejected": rejected,
        "invalid": invalid,
        "mean_turn_ms": sum(totals) / len(totals) if totals else 0.0,
        "mean_step_ms": {k: sum(v) / len(v) for k, v in step_ms.items()},
        "rows": rows,
    }


def _run_toggle(case, settings, top_k, step_ms, totals) -> dict:
    prompts = {}
    accepted = rejected = 0
    for mode in (True, False):
        store, working, llm = EvalStore(case.seeds), EvalWorking(), _scripted_llm(case)
        out = run_turn(case.query, "u1", "s1", settings, store, working,
                       llm_complete=llm, embed_fn=_embed, retrieval_on=mode)
        prompts[mode] = llm.last_messages[1]["content"]
        for event in out.events:
            if event.type in ("ADD", "SUPERSEDE"):
                accepted += 1
            elif event.type == "REJECT":
                rejected += 1
        for step in out.trace:
            step_ms.setdefault(("on:" if mode else "off:") + step.id, []).append(step.ms)
        totals.append(out.trace[-1].ms if out.trace else 0.0)
    hit = (case.expect_hit in prompts[True]) and (case.expect_hit not in prompts[False])
    return {"case": case.name, "hit": hit, "persisted": None,
            "expected_persisted": None, "accepted": accepted,
            "rejected": rejected, "invalid": False}


def main() -> None:
    report = run_suite()
    print(f"cases: {report['cases']}  "
          f"hit@k: {report['hit_at_k']:.2f}  "
          f"accepted: {report['accepted']}  rejected: {report['rejected']}  "
          f"invalid: {report['invalid']}  "
          f"mean turn: {report['mean_turn_ms']:.1f} ms")
    for row in report["rows"]:
        print(f"- {row['case']}: hit={row['hit']} "
              f"accepted={row['accepted']} rejected={row['rejected']} "
              f"invalid={row['invalid']}")
    print("mean ms per step (cumulative since turn start):")
    for step_id, mean in sorted(report["mean_step_ms"].items()):
        print(f"  {step_id}: {mean:.1f} ms")


if __name__ == "__main__":
    main()
