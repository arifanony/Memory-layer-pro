"""Environment configuration with startup validation (SCAFFOLD).

Reads plain environment variables, accepting the ADR's uppercase names as
well as lowercase ones (the operator's local .env uses lowercase).
`validate` fails fast with one actionable message instead of a deep stack
trace when the app cannot start.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_DATABASE_URL = "postgresql://user:password@localhost:5432/memory_lab"
DEFAULT_REDIS_URL = "redis://localhost:6379/0"
# ponytail: fixed 384-dim default so the pgvector schema has a known width;
# upgrade path is a documented migration, not a silent switch.
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_TOP_K = 3
DEFAULT_TTL_SECONDS = 3600


class ConfigError(Exception):
    """Raised when the environment cannot run the app. Message is user-facing."""


def _get(env: dict, name: str, default: str = "") -> str:
    for variant in (name.upper(), name.lower(), name):
        value = env.get(variant, "")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


def _read_dotenv(path: object = ".env") -> dict:
    """Minimal .env reader for local runs (KEY=VALUE, # comments, quotes).

    Only whole-file values; real environment variables always win. Never logs.
    """
    from pathlib import Path

    found: dict = {}
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return found
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value:
            found.setdefault(key, value)
    return found


@dataclass(frozen=True)
class Settings:
    nvidia_api_key: str
    nvidia_model: str
    nvidia_base_url: str = DEFAULT_BASE_URL
    database_url: str = DEFAULT_DATABASE_URL
    redis_url: str = DEFAULT_REDIS_URL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    top_k: int = DEFAULT_TOP_K
    redis_ttl_seconds: int = DEFAULT_TTL_SECONDS


def load(env: dict | None = None) -> Settings:
    """Build Settings from the environment, falling back to a local .env file
    next to this module. Use `load({})` in tests to skip the file entirely."""
    if env is None:
        from pathlib import Path

        src = _read_dotenv(Path(__file__).resolve().parent / ".env")
        src.update({k: v for k, v in os.environ.items() if v})
    else:
        src = dict(env)

    def _int(name: str, default: int) -> int:
        raw = _get(src, name, str(default))
        try:
            return int(raw)
        except ValueError:
            raise ConfigError(f"{name.upper()} must be an integer, got {raw!r}.")

    return Settings(
        nvidia_api_key=_get(src, "NVIDIA_API_KEY"),
        nvidia_model=_get(src, "NVIDIA_MODEL"),
        nvidia_base_url=_get(src, "NVIDIA_BASE_URL", DEFAULT_BASE_URL),
        database_url=_get(src, "DATABASE_URL", DEFAULT_DATABASE_URL),
        redis_url=_get(src, "REDIS_URL", DEFAULT_REDIS_URL),
        embedding_model=_get(src, "EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        top_k=_int("TOP_K", DEFAULT_TOP_K),
        redis_ttl_seconds=_int("REDIS_TTL_SECONDS", DEFAULT_TTL_SECONDS),
    )


def validate(settings: Settings) -> Settings:
    """Fail fast with actionable errors. Returns settings unchanged when valid."""
    problems = []
    if not settings.nvidia_api_key:
        problems.append("NVIDIA_API_KEY is missing: add it to .env (see .env.example).")
    if not settings.nvidia_model:
        problems.append(
            "NVIDIA_MODEL is missing: pick an id from the catalog "
            "(verified: nvidia/nemotron-3.5-lightning-30b-a3b) and set it in .env."
        )
    if settings.top_k < 1:
        problems.append(f"TOP_K must be >= 1, got {settings.top_k}.")
    if settings.redis_ttl_seconds < 1:
        problems.append(f"REDIS_TTL_SECONDS must be >= 1, got {settings.redis_ttl_seconds}.")
    for label, url in (("DATABASE_URL", settings.database_url), ("REDIS_URL", settings.redis_url)):
        if "://" not in url:
            problems.append(f"{label} looks invalid: {url!r}.")
    if problems:
        raise ConfigError("Invalid configuration:\n- " + "\n- ".join(problems))
    return settings
