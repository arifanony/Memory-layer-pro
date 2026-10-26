"""Local embeddings with one shared preprocessing path (EMBEDDINGS).

Both the query path and the memory path call `embed` below, so the vectors
are comparable by construction. The model id comes from Settings; the
384-dim width is asserted at load so a wrong model fails fast instead of
silently writing bad rows for MEMSTORE's pgvector column.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

from config import Settings

EXPECTED_DIM = 384


def cache_dir() -> Path:
    """Model cache. Project-local by default so gates, terminals, and CI share
    one location without touching the user's home directory."""
    override = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent / ".tools" / "hf-cache"

_WS = re.compile(r"\s+")


def prepare(text: str) -> str:
    """One normalization for every text, both paths. Strip, collapse runs."""
    return _WS.sub(" ", text).strip()


@lru_cache(maxsize=1)
def _load(model_id: str):  # type: ignore[no-untyped-def]
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id, cache_folder=str(cache_dir()))
    dim = model.get_sentence_embedding_dimension()
    if dim != EXPECTED_DIM:
        raise ValueError(
            f"Embedding model {model_id!r} has width {dim}, "
            f"expected {EXPECTED_DIM}; refusing to avoid corrupt pgvector rows."
        )
    return model


def embed(texts: list[str], settings: Settings) -> list[list[float]]:
    """Embed prepared texts. Empty input returns []. Whitespace-only → zeros? No:
    whitespace-only input is rejected as empty to avoid meaningless rows."""
    prepared = [prepare(text) for text in texts]
    if any(not item for item in prepared):
        raise ValueError("Cannot embed empty text; filter it before calling embed.")
    vectors = _load(settings.embedding_model).encode(prepared, normalize_embeddings=True)
    return [list(map(float, row)) for row in vectors]


def embed_query(text: str, settings: Settings) -> list[float]:
    """Single query path. Same prepare, same model as stored memories."""
    return embed([text], settings)[0]
