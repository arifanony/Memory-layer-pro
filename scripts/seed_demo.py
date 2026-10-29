"""Seed synthetic demo data, then print the full inventory.

Everything here is synthetic and safe to show on a projector. Re-running is
safe: memories whose text already exists for the demo user are skipped.
Run: .venv/bin/python scripts/seed_demo.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from config import load, validate
from embeddings import embed as embed_texts
from memory_store import PgStore, connect as pg_connect, init_schema
from schemas import MemoryEvent
from working_memory import RedisWorking, connect as redis_connect

DEMO_USER = "demo-user"
DEMO_SESSION = "demo-session"

# (text, category) — synthetic only, mirrors the layers in demo_guide.html.
SEEDS: list[tuple[str, str]] = [
    ("User prefers window seats when flying", "preference"),
    ("PostgreSQL pgvector extension enables cosine similarity search", "fact"),
    ("Redis lists with LTRIM keep only the newest N turns", "fact"),
    ("Use list comprehensions instead of map/filter for readable transforms",
     "procedure"),
    ("User takes coffee black in the morning", "preference"),
    ("Use collections.Counter for efficient frequency counting", "procedure"),
    ("User likes roast tea", "preference"),
]

WORKING_TURNS = [
    "user: what should I remember about my morning routine?",
    "assistant: You take coffee black in the morning and like roast tea.",
]


def main() -> int:
    settings = validate(load())
    pg_conn = pg_connect(settings)
    init_schema(pg_conn)
    store = PgStore(pg_conn)
    working = RedisWorking(redis_connect(settings), settings.redis_ttl_seconds)

    vectors = embed_texts([text for text, _ in SEEDS], settings)
    added = skipped = 0
    for (text, category), vector in zip(SEEDS, vectors):
        existing = {m.text for m in store.search(DEMO_USER, vector, 10)}
        if text in existing:
            skipped += 1
            continue
        mem = store.add(DEMO_USER, text, category, vector)
        store.record_event(DEMO_USER, MemoryEvent(type="ADD", memory_id=mem.id,
                                                 reason="synthetic seed"))
        added += 1
    print(f"seeded: {added} added, {skipped} already present")

    for turn in WORKING_TURNS:
        working.append(DEMO_USER, DEMO_SESSION, turn)

    print(f"\n--- durable memories for {DEMO_USER!r} ---")
    rows = pg_conn.execute(
        "SELECT text, category, created_at FROM memories "
        "WHERE user_id = %s ORDER BY created_at", (DEMO_USER,)).fetchall()
    for text, category, created in rows:
        print(f"[{category}] {text} (stored {created:%Y-%m-%d %H:%M})")

    print(f"\n--- memory events for {DEMO_USER!r} (latest 10) ---")
    events = pg_conn.execute(
        "SELECT type, reason FROM memory_events "
        "WHERE user_id = %s ORDER BY id DESC LIMIT 10", (DEMO_USER,)).fetchall()
    for type_, reason in events:
        print(f"{type_}: {reason}")

    print(f"\n--- working context wm:{DEMO_USER}:{DEMO_SESSION} ---")
    for turn in working.read(DEMO_USER, DEMO_SESSION, 10):
        print(f"  {turn}")
    pg_conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
