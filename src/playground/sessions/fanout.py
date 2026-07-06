from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

from playground.db.models import ModelThread
from playground.ids import encode
from playground.runtime.client import AgentRuntimeClient


async def fanout_chat(
    runtime: AgentRuntimeClient,
    threads: list[tuple[ModelThread, str | None]],
    user_message: str,
    tools: list[str] | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Merge N per-thread runtime streams into one internal event stream."""
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def _pump(thread: ModelThread, reasoning_effort: str | None) -> None:
        thread_id = encode(thread.id)
        try:
            await queue.put(
                {
                    "type": "thread_start",
                    "thread_id": thread_id,
                    "provider": thread.provider,
                    "model": thread.model_name,
                }
            )

            async for event in runtime.chat_stream(
                thread.runtime_session_id,
                user_message,
                provider=thread.provider,
                model=thread.model_name,
                reasoning_effort=reasoning_effort,
                tools=tools,
            ):
                event["thread_id"] = thread_id
                await queue.put(event)
        except Exception as exc:
            await queue.put(
                {
                    "type": "error",
                    "thread_id": thread_id,
                    "error": str(exc),
                }
            )
        finally:
            await queue.put({"type": "_thread_finished", "thread_id": thread_id})

    tasks = [
        asyncio.create_task(_pump(thread, reasoning_effort))
        for thread, reasoning_effort in threads
    ]
    remaining = len(tasks)

    while remaining > 0:
        item = await queue.get()
        if item.get("type") == "_thread_finished":
            remaining -= 1
            continue
        yield item

    while not queue.empty():
        item = queue.get_nowait()
        if item.get("type") != "_thread_finished":
            yield item
