"""Async HTTP client for the agent runtime service."""

import json
from collections.abc import AsyncGenerator

import httpx

import playground.ids as ids
from playground.config import get_settings


class AgentRuntimeClient:
    """Thin async wrapper around the agent-runtime HTTP API.

    The underlying httpx client is created lazily on first use.
    Call ``close()`` when done (or use as an async context manager).
    """

    def __init__(self, base_url: str | None = None) -> None:
        self._base_url = (base_url or get_settings().agent_runtime_url).rstrip("/")
        self._client: httpx.AsyncClient | None = None

    # -- lifecycle -----------------------------------------------------------

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(300),
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "AgentRuntimeClient":
        return self

    async def __aexit__(self, *exc) -> None:  # noqa: ANN002
        await self.close()

    # -- public API ----------------------------------------------------------

    async def create_session(self, provider: str, model_name: str) -> str:
        """Create a new runtime session and return its encoded ID."""
        resp = await self.client.post(
            "/sessions",
            json={"provider": provider, "model_name": model_name},
        )
        resp.raise_for_status()
        data = resp.json()
        return ids.encode(data["session_id"])

    async def chat_stream(
        self, session_id: str, message: str
    ) -> AsyncGenerator[dict, None]:
        """Stream SSE events from a chat completion request.

        Yields parsed JSON objects from each ``data:`` line.
        """
        url = f"/sessions/{session_id}/chat/stream"
        async with self.client.stream(
            "POST",
            url,
            json={"message": message},
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
