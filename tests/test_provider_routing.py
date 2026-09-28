"""Verify isolated provider credentials and the real SDK request format offline."""
import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.config import get_blue_provider, provider_client_kwargs
from core.openai_runtime import OpenAIRunner, create_blue_pair, create_openai_pair


@pytest.mark.parametrize("factory,provider,url,role", [
    (create_blue_pair, "mistral", "https://api.mistral.ai/v1/chat/completions", "system"),
    (create_openai_pair, "cohere", "https://api.cohere.ai/compatibility/v1/chat/completions", "developer"),
])
def test_sdk_routes_each_team_to_its_own_key_endpoint_and_instruction(
    monkeypatch, factory, provider, url, role
):
    monkeypatch.setenv("BLUE_PROVIDER", "mistral")
    monkeypatch.setenv("RED_TEAM_PROVIDER", "cohere")
    monkeypatch.setenv("BLUE_MODEL", "blue-test-model")
    monkeypatch.setenv("RED_MODEL", "red-test-model")
    for p in ("mistral", "cohere", "openai"):
        monkeypatch.setenv(f"{p.upper()}_API_KEY", f"fake-{p}-test-key")
        monkeypatch.delenv(f"{p.upper()}_BASE_URL", raising=False)
    sent = []

    def handler(request):
        assert str(request.url) == url
        assert request.headers["Authorization"] == f"Bearer fake-{provider}-test-key"
        body = json.loads(request.content)
        assert body["model"] == ("blue-test-model" if provider == "mistral" else "red-test-model")
        assert body["messages"] == [
            {"role": role, "content": "test instruction"},
            {"role": "user", "content": "banking question"},
        ]
        sent.append(body)
        return httpx.Response(200, json={
            "id": "test", "object": "chat.completion", "created": 0,
            "model": body["model"], "choices": [{"index": 0,
                "message": {"role": "assistant", "content": "test reply"},
                "finish_reason": "stop"}],
        })

    monkeypatch.setattr(OpenAIRunner, "_client", lambda self: OpenAI(
        max_retries=0, **self.client_kwargs,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    ))
    agent, runner = factory(name="test", instruction="test instruction", app_name="test")
    assert asyncio.run(runner.chat(agent, "banking question")) == "test reply"
    assert len(sent) == 1


def test_missing_selected_key_never_falls_back_to_another_provider(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.setenv("COHERE_API_KEY", "fake-test-key")
    with pytest.raises(RuntimeError, match="Missing MISTRAL_API_KEY"):
        provider_client_kwargs("mistral")


def test_invalid_provider_fails_before_credentials_can_be_routed(monkeypatch):
    monkeypatch.setenv("BLUE_PROVIDER", "unknown-provider")
    with pytest.raises(ValueError, match="Unsupported provider"):
        get_blue_provider()
