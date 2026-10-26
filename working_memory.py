"""Redis working memory: bounded, expiring, never durable (WORKMEM).

One list per session (`wm:{session_id}`), newest last. Every append trims to
MAX_TURNS and refreshes the TTL, so idle sessions evaporate and hot ones stay
bounded. This store is explicitly not durable: restarts and evictions may lose
turns, and nothing here is ever promoted to long-term memory.
"""

from __future__ import annotations

import redis

from config import Settings

MAX_TURNS = 50


def _wrap(label: str):
    """Translate driver errors to OSError for the orchestrator's handling."""

    def decorator(fn):  # type: ignore[no-untyped-def]
        def inner(*args, **kwargs):  # type: ignore[no-untyped-def]
            try:
                return fn(*args, **kwargs)
            except redis.RedisError as exc:
                raise OSError(f"redis working memory {label} failed: {exc}") from exc

        return inner

    return decorator


def connect(settings: Settings) -> redis.Redis:
    return redis.Redis.from_url(settings.redis_url, decode_responses=True)


class RedisWorking:
    """WorkingMemory on Redis. Raises OSError (never driver errors)."""

    def __init__(self, client: redis.Redis, ttl_seconds: int):
        self.client = client
        self.ttl_seconds = ttl_seconds

    @_wrap("read")
    def read(self, session_id: str, limit: int) -> list[str]:
        if limit < 1:
            return []
        return self.client.lrange(f"wm:{session_id}", -limit, -1)

    @_wrap("append")
    def append(self, session_id: str, turn: str) -> None:
        key = f"wm:{session_id}"
        pipe = self.client.pipeline()
        pipe.rpush(key, turn)
        pipe.ltrim(key, -MAX_TURNS, -1)
        pipe.expire(key, self.ttl_seconds)
        pipe.execute()

    @_wrap("clear")
    def clear(self, session_id: str) -> None:
        self.client.delete(f"wm:{session_id}")
