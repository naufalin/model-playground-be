from __future__ import annotations

import json

import httpx

from playground.config import get_settings
from playground.runtime.client import AgentRuntimeClient


def _runtime_with_transport(handler) -> AgentRuntimeClient:
    runtime = AgentRuntimeClient(base_url="http://runtime")
    runtime._client = httpx.AsyncClient(  # noqa: SLF001
        base_url="http://runtime",
        transport=httpx.MockTransport(handler),
        headers=runtime._headers,  # noqa: SLF001
    )
    return runtime


async def test_runtime_client_sends_bearer_and_keeps_encoded_session_id(monkeypatch) -> None:
    monkeypatch.setenv("SECRET_KEY", "test")
    monkeypatch.setenv("AGENT_RUNTIME_BEARER_TOKEN", "runtime-token")
    get_settings.cache_clear()

    seen_headers = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers["authorization"] = request.headers.get("authorization")
        assert request.url.path == "/sessions"
        assert json.loads(request.content) == {"title": "openrouter/model", "tools": None}
        return httpx.Response(201, json={"id": "encoded-runtime-id"})

    runtime = _runtime_with_transport(handler)
    try:
        session_id = await runtime.create_session(title="openrouter/model")
    finally:
        await runtime.close()

    assert seen_headers["authorization"] == "Bearer runtime-token"
    assert session_id == "encoded-runtime-id"


async def test_runtime_client_chat_stream_passes_model_options(monkeypatch) -> None:
    monkeypatch.setenv("SECRET_KEY", "test")
    monkeypatch.setenv("AGENT_RUNTIME_BEARER_TOKEN", "runtime-token")
    get_settings.cache_clear()

    seen_payload = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_payload.update(json.loads(request.content))
        assert request.url.path == "/sessions/runtime-1/chat/stream"
        body = 'data: {"type":"text_delta","delta":"hi"}\n\n'
        body += 'data: {"type":"done","usage":{"total_tokens":1}}\n\n'
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    runtime = _runtime_with_transport(handler)
    try:
        events = [
            event
            async for event in runtime.chat_stream(
                "runtime-1",
                "hello",
                provider="openrouter",
                model="vendor/model",
                reasoning_effort="high",
                tools=["web_search"],
            )
        ]
    finally:
        await runtime.close()

    assert seen_payload == {
        "message": "hello",
        "provider": "openrouter",
        "model": "vendor/model",
        "reasoning_effort": "high",
        "tools": ["web_search"],
    }
    assert events[-1]["type"] == "done"


async def test_runtime_client_lists_tools(monkeypatch) -> None:
    monkeypatch.setenv("SECRET_KEY", "test")
    get_settings.cache_clear()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/tools"
        return httpx.Response(200, json={"tools": [{"name": "web_search"}], "total": 1})

    runtime = _runtime_with_transport(handler)
    try:
        payload = await runtime.list_tools()
    finally:
        await runtime.close()

    assert payload == {"tools": [{"name": "web_search"}], "total": 1}


async def test_runtime_client_lists_models(monkeypatch) -> None:
    monkeypatch.setenv("SECRET_KEY", "test")
    get_settings.cache_clear()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/models"
        return httpx.Response(
            200,
            json={
                "default_provider": "openrouter",
                "openrouter": {"models": [{"model_id": "vendor/model"}]},
            },
        )

    runtime = _runtime_with_transport(handler)
    try:
        payload = await runtime.list_models()
    finally:
        await runtime.close()

    assert payload["default_provider"] == "openrouter"
    assert payload["openrouter"]["models"][0]["model_id"] == "vendor/model"


async def test_runtime_client_creates_model(monkeypatch) -> None:
    monkeypatch.setenv("SECRET_KEY", "test")
    get_settings.cache_clear()

    seen_payload = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/models"
        seen_payload.update(json.loads(request.content))
        return httpx.Response(201, json={"id": 10, **seen_payload})

    runtime = _runtime_with_transport(handler)
    try:
        payload = await runtime.create_model(
            provider="openrouter",
            model_id="vendor/new",
            name="Vendor New",
            enabled=True,
            supports_reasoning=False,
            sort_order=80,
            config={"tier": "test"},
        )
    finally:
        await runtime.close()

    assert seen_payload == {
        "provider": "openrouter",
        "model_id": "vendor/new",
        "name": "Vendor New",
        "enabled": True,
        "supports_reasoning": False,
        "sort_order": 80,
        "config": {"tier": "test"},
    }
    assert payload["id"] == 10
