"""OpenRouter provider + Nemotron Ultra option (LLMCLIENT/CONFIG)."""

from config import load, validate
from llm_client import build_request


def test_dotenv_reader_strips_export_prefix(tmp_path):
    from config import _read_dotenv

    path = tmp_path / ".env"
    path.write_text('export OPENROUTER_API_KEY="sk-test"\n')
    assert _read_dotenv(path) == {"OPENROUTER_API_KEY": "sk-test"}


def test_openrouter_auto_selected_with_ultra_default():
    settings = validate(load({"OPENROUTER_API_KEY": "sk-test"}))
    assert settings.llm_provider == "openrouter"
    assert settings.openrouter_model == "nvidia/llama-3.1-nemotron-ultra-253b-v1:free"
    assert settings.openrouter_base_url.startswith("https://")


def test_build_request_uses_openrouter_endpoint_and_headers():
    import json

    settings = validate(load({"OPENROUTER_API_KEY": "sk-test"}))
    request = build_request([{"role": "user", "content": "hi"}], settings)
    assert request.full_url == settings.openrouter_base_url.rstrip("/") + "/chat/completions"
    sent = json.loads(request.data.decode())
    assert sent["model"] == "nvidia/llama-3.1-nemotron-ultra-253b-v1:free"
    assert request.headers.get("Authorization") == "Bearer sk-test"
