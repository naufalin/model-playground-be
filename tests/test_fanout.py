from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from playground.sessions.fanout import fanout_chat


@pytest.mark.asyncio
async def test_closing_fanout_cancels_each_runtime_stream() -> None:
    cancelled = [asyncio.Event(), asyncio.Event()]

    class BlockingRuntime:
        async def chat_stream(self, session_id: str, message: str, **kwargs):
            index = int(session_id.removeprefix("runtime-"))
            try:
                await asyncio.Event().wait()
                yield
            finally:
                cancelled[index].set()

    threads = [
        (
            SimpleNamespace(
                id=index + 1,
                runtime_session_id=f"runtime-{index}",
                provider="openai",
                model_name=f"model-{index}",
            ),
            None,
        )
        for index in range(2)
    ]
    stream = fanout_chat(BlockingRuntime(), threads, "hello")

    first_event = await anext(stream)
    assert first_event["type"] == "thread_start"

    await stream.aclose()

    assert all(event.is_set() for event in cancelled)
