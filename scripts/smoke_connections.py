"""Connection smoke test: verifies every configured URL is reachable.

Checks NVIDIA + OpenRouter (list/key endpoints only — never a chat
completion, so it spends no tokens), plus Postgres and Redis round-trips.
Run: .venv/bin/python scripts/smoke_connections.py
"""

from __future__ import annotations

import json
import sys
import urllib.request

sys.path.insert(0, ".")

from config import load, validate  # noqa: E402


def _get(url: str, headers: dict, timeout: int = 15) -> tuple[int | str, str]:
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(200).decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001 — report, don't raise
        return type(exc).__name__, str(exc)[:160]


def main() -> int:
    settings = validate(load())
    ok = True

    # NVIDIA: model catalog proves key + endpoint + network in one call.
    status, body = _get(
        settings.nvidia_base_url.rstrip("/") + "/models",
        {"Authorization": f"Bearer {settings.nvidia_api_key}"},
    )
    print(f"NVIDIA {settings.nvidia_base_url}: {status} {body[:80]}")
    ok &= status == 200

    # OpenRouter: key endpoint reports key validity, spends nothing.
    if getattr(settings, "openrouter_api_key", ""):
        base = getattr(settings, "openrouter_base_url",
                       "https://openrouter.ai/api/v1").rstrip("/")
        status, body = _get(
            base + "/auth/key",
            {"Authorization": f"Bearer {settings.openrouter_api_key}"},
        )
        try:
            label = json.loads(body).get("data", {}).get("label", "")
        except (ValueError, AttributeError):
            label = ""
        print(f"OpenRouter {base}: {status} {label or body[:80]}")
        ok &= status == 200
    else:
        print("OpenRouter: skipped (no key set)")

    # Postgres + pgvector.
    try:
        from memory_store import connect as pg_connect

        conn = pg_connect(settings)
        version = conn.execute("SELECT version()").fetchone()[0].split(",")[0]
        vector = conn.execute(
            "SELECT extversion FROM pg_extension WHERE extname='vector'"
        ).fetchone()
        print(f"Postgres: OK {version} pgvector={vector[0] if vector else 'MISSING'}")
        ok &= vector is not None
        conn.close()
    except Exception as exc:  # noqa: BLE001
        print(f"Postgres: FAIL {type(exc).__name__}: {str(exc)[:120]}")
        ok = False

    # Redis.
    try:
        from working_memory import connect as redis_connect

        client = redis_connect(settings)
        print(f"Redis: {'OK' if client.ping() else 'FAIL ping false'}")
        ok &= client.ping()
    except Exception as exc:  # noqa: BLE001
        print(f"Redis: FAIL {type(exc).__name__}: {str(exc)[:120]}")
        ok = False

    print("ALL OK" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
