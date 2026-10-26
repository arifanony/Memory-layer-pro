"""UI presentation helpers and redaction guarantees (UI; FR-9/10, NFR-4, AC-7/8)."""

import app
from app import event_line, failed_steps, format_ms, memory_line, status_label
from orchestrator import TraceStep
from schemas import MemoryEvent, StoredMemory


def test_format_ms_readable():
    assert format_ms(0.02) == "20.0 ms"
    assert format_ms(1.5) == "1500 ms"


def test_status_labels_plain_words():
    assert status_label("ok") == "done"
    assert status_label("failed") == "failed"
    assert status_label("skipped") == "skipped"


def test_memory_line_shows_category_and_score():
    line = memory_line(StoredMemory(id="m1", text="likes tea", category="preference", score=0.91))
    assert "preference" in line and "likes tea" in line and "0.91" in line


def test_event_line_links_supersede():
    line = event_line(MemoryEvent(type="SUPERSEDE", reason="contradicts", supersedes="m1"))
    assert "SUPERSEDE" in line and "m1" in line


def test_failed_steps_filter():
    trace = [TraceStep(id="a", label="A", status="ok"), TraceStep(id="b", label="B", status="failed")]
    assert [s.id for s in failed_steps(trace)] == ["b"]


def test_app_module_import_has_no_secrets():
    import pathlib
    source = pathlib.Path(app.__file__).read_text()
    assert "sk-" not in source and "BEGIN" not in source
