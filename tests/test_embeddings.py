"""Local embeddings: shared path, width, determinism (EMBEDDINGS; FR-4, NFR-2, AC-4)."""

import pytest

from config import Settings, load, validate
from embeddings import EXPECTED_DIM, embed, embed_query, prepare


@pytest.fixture(scope="module")
def settings() -> Settings:
    return validate(load({"NVIDIA_API_KEY": "k", "NVIDIA_MODEL": "m"}))


def test_prepare_is_shared_and_trivial():
    assert prepare("  hello   world\n") == "hello world"


def test_width_matches_pgvector_contract(settings: Settings):
    assert len(embed_query("hello world", settings)) == EXPECTED_DIM == 384


def test_query_and_memory_paths_agree(settings: Settings):
    assert embed_query("hello world", settings) == embed(["hello world"], settings)[0]


def test_deterministic(settings: Settings):
    assert embed(["stable input"], settings) == embed(["stable input"], settings)


def test_empty_rejected(settings: Settings):
    with pytest.raises(ValueError, match="[Ee]mpty"):
        embed(["   "], settings)
