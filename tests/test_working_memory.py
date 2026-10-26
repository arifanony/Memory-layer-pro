"""Redis working memory: bounds, TTL, clear, keying (WORKMEM; FR-1/3/8, AC-2/9).

fakeredis proves the logic everywhere. Live tests need project Redis on
localhost:6379 and skip otherwise — run them where services are up.
"""

import pytest
import redis

from config import validate, load
from working_memory import MAX_TURNS, RedisWorking, connect


def _settings(**overrides):
    env = {"NVIDIA_API_KEY": "k", "NVIDIA_MODEL": "m", "REDIS_URL": "redis://localhost:6379/0"}
    env.update(overrides)
    return validate(load(env))


def _live():
    try:
        client = connect(_settings())
        client.ping()
        return client
    except redis.RedisError:
        pytest.skip("project Redis unavailable; run scripts/setup-services.sh")


def _fake(ttl=60):
    import fakeredis
    return RedisWorking(fakeredis.FakeRedis(decode_responses=True), ttl)


def test_append_read_roundtrip_in_order():
    wm = _fake()
    wm.append("s1", "user: hi")
    wm.append("s1", "assistant: hello")
    assert wm.read("s1", 10) == ["user: hi", "assistant: hello"]


def test_read_respects_limit_newest_last():
    wm = _fake()
    for i in range(5):
        wm.append("s1", f"t{i}")
    assert wm.read("s1", 2) == ["t3", "t4"]


def test_history_bounded_to_max_turns():
    wm = _fake()
    for i in range(MAX_TURNS + 10):
        wm.append("s1", f"t{i}")
    assert len(wm.read("s1", MAX_TURNS + 10)) == MAX_TURNS


def test_sessions_are_isolated_keys():
    wm = _fake()
    wm.append("s1", "one")
    wm.append("s2", "two")
    assert wm.read("s1", 10) == ["one"]
    assert wm.read("s2", 10) == ["two"]


def test_clear_session_empties_only_that_session():
    wm = _fake()
    wm.append("s1", "one")
    wm.append("s2", "two")
    wm.clear("s1")
    assert wm.read("s1", 10) == []
    assert wm.read("s2", 10) == ["two"]


def test_ttl_applied_on_append():
    import fakeredis
    client = fakeredis.FakeRedis(decode_responses=True)
    RedisWorking(client, 123).append("s1", "hi")
    assert client.ttl("wm:s1") == 123


def test_driver_errors_become_oserror():
    class BadClient:
        def pipeline(self):
            raise redis.ConnectionError("down")

    with pytest.raises(OSError, match="redis working memory append failed"):
        RedisWorking(BadClient(), 60).append("s1", "hi")


def test_live_ttl_and_clear():
    client = _live()
    wm = RedisWorking(client, 60)
    wm.append("live-s", "hello")
    assert 0 < client.ttl("wm:live-s") <= 60
    wm.clear("live-s")
    assert wm.read("live-s", 10) == []
