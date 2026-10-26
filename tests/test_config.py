"""Config loading and validation (SCAFFOLD, NFR-6/AC-1)."""

import pytest

from config import ConfigError, _read_dotenv, load, validate


def test_defaults_apply():
    settings = validate(load({"NVIDIA_API_KEY": "k", "NVIDIA_MODEL": "m"}))
    assert settings.nvidia_base_url.startswith("https://")
    assert settings.top_k == 3
    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"


def test_lowercase_env_accepted():
    settings = load({"nvidia_api_key": "k", "nvidia_model": "m"})
    assert settings.nvidia_api_key == "k"


def test_missing_key_fails_actionably():
    with pytest.raises(ConfigError, match="NVIDIA_API_KEY"):
        validate(load({"NVIDIA_MODEL": "m"}))


def test_missing_model_fails_actionably():
    with pytest.raises(ConfigError, match="NVIDIA_MODEL"):
        validate(load({"NVIDIA_API_KEY": "k"}))


def test_bad_int_rejected():
    with pytest.raises(ConfigError, match="TOP_K"):
        load({"NVIDIA_API_KEY": "k", "NVIDIA_MODEL": "m", "TOP_K": "many"})


def test_dotenv_reader_ignores_comments_and_empty(tmp_path):
    path = tmp_path / ".env"
    path.write_text('# comment\n\nnvidia_api_key = "abc"\nEMPTY=\n')
    assert _read_dotenv(path) == {"nvidia_api_key": "abc"}


def test_dotenv_missing_file_returns_empty(tmp_path):
    assert _read_dotenv(tmp_path / "nope.env") == {}
