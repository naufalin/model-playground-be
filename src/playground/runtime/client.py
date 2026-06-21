"""Async HTTP client for the agent runtime service."""

import json
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from playground.config import get_settings


class AgentRuntimeClient:
    """Thin async wrapper around the agent-runtime HTTP API.

    The underlying httpx client is created lazily on first use.
    Call ``close()`` when done (or use as an async context manager).
    """

    def __init__(self, base_url: str | None = None) -> None:
        settings = get_settings()
        self._base_url = (base_url or settings.agent_runtime_url).rstrip("/")
        self._bearer_token = settings.agent_runtime_bearer_token.strip()
        self._client: httpx.AsyncClient | None = None

    # -- lifecycle -----------------------------------------------------------

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(300),
                headers=self._headers,
            )
        return self._client

    @property
    def _headers(self) -> dict[str, str]:
        if not self._bearer_token:
            return {}
        return {"Authorization": f"Bearer {self._bearer_token}"}

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "AgentRuntimeClient":
        return self

    async def __aexit__(self, *exc) -> None:  # noqa: ANN002
        await self.close()

    # -- public API ----------------------------------------------------------

    async def list_models(self) -> dict[str, Any]:
        """Return the runtime model registry payload."""
        resp = await self.client.get("/models")
        resp.raise_for_status()
        return resp.json()

    async def create_model(
        self,
        *,
        provider: str,
        model_id: str,
        name: str,
        enabled: bool = True,
        supports_reasoning: bool = False,
        sort_order: int = 0,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a runtime model and return the created model payload."""
        payload: dict[str, Any] = {
            "provider": provider,
            "model_id": model_id,
            "name": name,
            "enabled": enabled,
            "supports_reasoning": supports_reasoning,
            "sort_order": sort_order,
            "config": config,
        }
        resp = await self.client.post("/models", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def create_session(self, title: str = "New Session") -> str:
        """Create a new runtime session and return its encoded ID."""
        resp = await self.client.post(
            "/sessions",
            json={"title": title},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["id"]

    async def chat_stream(
        self,
        session_id: str,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> AsyncGenerator[dict, None]:
        """Stream SSE events from a chat completion request.

        Yields parsed JSON objects from each ``data:`` line.
        """
        url = f"/sessions/{session_id}/chat/stream"
        payload: dict[str, Any] = {"message": message}
        if provider is not None:
            payload["provider"] = provider
        if model is not None:
            payload["model"] = model
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort
        async with self.client.stream(
            "POST",
            url,
            json=payload,
            headers={"Accept": "text/event-stream"},
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    payload = line[6:]  # strip 'data: ' prefix
                    if payload.strip() == "[DONE]":
                        return
                    yield json.loads(payload)
