"""pgvector store: scoping, schema, events (MEMSTORE; FR-5/7/13, NFR-3, AC-2/5/12).

Logic tests run everywhere against a recording fake connection. Live tests
need the project Postgres on localhost:5432 and skip honestly otherwise —
run them in a terminal with services up: ./scripts/setup-services.sh
"""

import psycopg
import pytest

from config import validate, load
from memory_store import DIM, PgStore, SCHEMA_SQL, connect, init_schema
from schemas import MemoryEvent


def _settings(**overrides):
    env = {"NVIDIA_API_KEY": "k", "NVIDIA_MODEL": "m"}
    env.update(overrides)
    return validate(load(env))


class FakeCur:
    def __init__(self):
        self.calls = []
        self.rows = []

    def execute(self, sql, params=()):
        self.calls.append((sql, params))
        return self

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.rows[0] if self.rows else None


class FakeConn:
    def __init__(self, rows=()):
        self.cur = FakeCur()
        self.cur.rows = list(rows)

    def execute(self, sql, params=()):
        return self.cur.execute(sql, params)


def _live():
    try:
        conn = psycopg.connect("dbname=memory_lab user=postgres host=localhost port=5432",
                               autocommit=True)
        conn.execute("SELECT 1")
        return conn
    except psycopg.Error:
        pytest.skip("project Postgres unavailable; run scripts/setup-services.sh")


def test_schema_creates_user_scoped_tables():
    assert "CREATE EXTENSION IF NOT EXISTS vector" in SCHEMA_SQL
    assert "embedding vector(384)" in SCHEMA_SQL
    assert "memory_events" in SCHEMA_SQL
    assert DIM == 384


def test_search_is_user_scoped_and_bounded():
    conn = FakeConn([("id1", "t", "fact", 0.9)])
    store = PgStore(conn)
    out = store.search("u1", [0.1] * 384, 3)
    sql, params = conn.cur.calls[0]
    assert "user_id = %s" in sql and "LIMIT %s" in sql
    assert params[1] == "u1" and params[3] == 3
    assert out[0].score == pytest.approx(0.9)


def test_add_returns_row_with_user_scope():
    conn = FakeConn([("new-id",)])
    mem = PgStore(conn).add("u9", "likes tea", "preference", [0.0] * 384)
    sql, params = conn.cur.calls[0]
    assert "INSERT INTO memories" in sql and params[:3] == ("u9", "likes tea", "preference")
    assert mem.id == "new-id" and mem.text == "likes tea"


def test_record_event_persists_type_and_reason():
    conn = FakeConn()
    PgStore(conn).record_event("u1", MemoryEvent(type="REJECT", reason="dup"))
    sql, params = conn.cur.calls[0]
    assert "memory_events" in sql and params[2:] == ("REJECT", "dup")


def test_forget_user_deletes_scoped_rows():
    class CountingCur(FakeCur):
        def __init__(self):
            super().__init__()
            self.rowcount = 2

    conn = FakeConn()
    conn.cur = CountingCur()
    out = PgStore(conn).forget_user("u1")
    sqls = [call[0] for call in conn.cur.calls]
    assert any("DELETE FROM memory_events" in s for s in sqls)
    assert any("DELETE FROM memories" in s for s in sqls)
    assert all(call[1] == ("u1",) for call in conn.cur.calls)
    assert out == {"memories": 2, "events": 2}


def test_driver_errors_become_oserror():
    class BadConn:
        def execute(self, *a, **k):
            raise psycopg.OperationalError("down")

    with pytest.raises(OSError, match="pgvector store search failed"):
        PgStore(BadConn()).search("u1", [0.0] * 384, 1)


def test_live_roundtrip_add_search_event():
    conn = _live()
    init_schema(conn)
    store = PgStore(conn)
    vec = [0.05] * 384
    mem = store.add("live-u", "lives in Paris", "fact", vec)
    try:
        hits = store.search("live-u", vec, 3)
        assert hits and hits[0].id == mem.id and hits[0].score > 0.99
        assert store.search("other-u", vec, 3) == []
        store.record_event("live-u", MemoryEvent(type="ADD", reason="t", memory_id=mem.id))
        n = conn.execute("SELECT count(*) FROM memory_events WHERE user_id='live-u'").fetchone()[0]
        assert n >= 1
        with pytest.raises(OSError):
            store.add("live-u", "bad dim", "fact", [0.0] * 3)
    finally:
        conn.execute("DELETE FROM memory_events WHERE user_id='live-u'")
        conn.execute("DELETE FROM memories WHERE user_id='live-u'")


def test_live_timestamps_present():
    conn = _live()
    init_schema(conn)
    row = conn.execute(
        "SELECT created_at IS NOT NULL, updated_at IS NOT NULL, valid_from IS NOT NULL"
        " FROM memories LIMIT 1").fetchone()
    assert row is None or all(row)
