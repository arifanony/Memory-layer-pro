"""PostgreSQL + pgvector durable store (MEMSTORE).

Every read, write, and delete is scoped by user_id. Vectors are 384-wide
HNSW-indexed cosine embeddings matching embeddings.EXPECTED_DIM. Event rows
mirror every orchestrator decision so the trace is backed by durable proof.

Schema changes: never ALTER the vector width in place. Add a new column,
backfill, then swap readers — see UPGRADE below.
"""

from __future__ import annotations

import psycopg
from pgvector.psycopg import register_vector

from config import Settings
from schemas import MemoryEvent, StoredMemory

DIM = 384

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    text TEXT NOT NULL,
    category TEXT NOT NULL,
    embedding vector(384) NOT NULL,
    supersedes UUID REFERENCES memories(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    valid_from TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS memories_embedding_idx
    ON memories USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS memories_user_idx ON memories (user_id);
CREATE TABLE IF NOT EXISTS memory_events (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    memory_id UUID REFERENCES memories(id),
    type TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS memory_events_user_idx ON memory_events (user_id);
-- UPGRADE: to change DIM, add embedding_v2 vector(NEW), backfill from the
-- model, then point search/add at the new column and drop the old one.
"""


def connect(settings: Settings) -> psycopg.Connection:
    """Open an autocommit connection. Caller owns close()."""
    conn = psycopg.connect(settings.database_url, autocommit=True)
    register_vector(conn)
    return conn


def init_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA_SQL)


def _wrap(label: str):
    """Translate driver errors to OSError so the orchestrator's failure
    handling (and its fakes) stays uniform."""

    def decorator(fn):  # type: ignore[no-untyped-def]
        def inner(*args, **kwargs):  # type: ignore[no-untyped-def]
            try:
                return fn(*args, **kwargs)
            except psycopg.Error as exc:
                raise OSError(f"pgvector store {label} failed: {exc}") from exc

        return inner

    return decorator


class PgStore:
    """MemoryStore on pgvector. Methods raise OSError (never driver errors)."""

    def __init__(self, conn: psycopg.Connection):
        self.conn = conn

    @_wrap("search")
    def search(self, user_id: str, vector: list[float], top_k: int) -> list[StoredMemory]:
        rows = self.conn.execute(
            "SELECT id, text, category, 1 - (embedding <=> %s::vector) AS score"
            " FROM memories WHERE user_id = %s"
            " ORDER BY embedding <=> %s::vector LIMIT %s",
            (vector, user_id, vector, top_k),
        ).fetchall()
        return [StoredMemory(id=str(row[0]), text=row[1], category=row[2],
                             score=float(row[3])) for row in rows]

    @_wrap("add")
    def add(self, user_id: str, text: str, category: str, vector: list[float]) -> StoredMemory:
        row = self.conn.execute(
            "INSERT INTO memories (user_id, text, category, embedding)"
            " VALUES (%s, %s, %s, %s) RETURNING id",
            (user_id, text, category, vector),
        ).fetchone()
        return StoredMemory(id=str(row[0]), text=text, category=category)

    @_wrap("link_supersede")
    def link_supersede(self, old_id: str, new_id: str) -> None:
        self.conn.execute(
            "UPDATE memories SET supersedes = %s, updated_at = now() WHERE id = %s",
            (old_id, new_id),
        )

    @_wrap("record_event")
    def record_event(self, user_id: str, event: MemoryEvent) -> None:
        self.conn.execute(
            "INSERT INTO memory_events (user_id, memory_id, type, reason)"
            " VALUES (%s, %s, %s, %s)",
            (user_id, event.memory_id, event.type, event.reason),
        )
