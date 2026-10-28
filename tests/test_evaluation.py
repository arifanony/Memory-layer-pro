"""Deterministic synthetic eval set (FR-12): no live key, services, or network."""

from evaluation import default_cases, run_suite


def test_suite_needs_no_live_key_and_scores_perfectly():
    report = run_suite()
    assert report["cases"] == len(default_cases()) == 6
    assert report["hit_at_k"] == 1.0
    assert report["invalid"] == 1
    assert report["accepted"] == 0
    assert report["rejected"] == 1


def test_rejected_and_invalid_cases_persist_nothing():
    report = run_suite()
    rows = {r["case"]: r for r in report["rows"]}
    assert rows["policy_rejects_pii"]["persisted"] is False
    assert rows["invalid_output"]["persisted"] is False
    assert rows["ephemeral_not_persisted"]["persisted"] is False


def test_retrieval_toggle_excludes_memories_from_prompt():
    report = run_suite()
    rows = {r["case"]: r for r in report["rows"]}
    assert rows["retrieval_toggle"]["hit"] is True


def test_latency_reported_per_stage():
    report = run_suite()
    assert report["mean_turn_ms"] >= 0.0
    assert report["mean_step_ms"], "expected per-step latency means"
    assert "model_call" in report["mean_step_ms"]
