from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator

from playground.db.models import ModelThread
from playground.ids import encode
from playground.runtime.client import AgentRuntimeClient


async def fanout_chat(
    runtime: AgentRuntimeClient,
    threads: list[ModelThread],
    user_message: str,
) -> AsyncGenerator[str, None]:
    """Merge N per-thread streaming responses into a single SSE stream."""
    queue: asyncio.Queue[str] = asyncio.Queue()

    async def _pump(thread: ModelThread) -> None:
        thread_id = encode(thread.id)
        try:
            await queue.put(json.dumps({"type": "thread_start", "thread_id": thread_id}))

            full_text = ""
            start = time.monotonic()
            async for event in runtime.chat_stream(thread.runtime_session_id, user_message):
                event["thread_id"] = thread_id
                await queue.put(json.dumps(event))
                if event.get("type") == "text_delta":
                    full_text += event.get("delta", "")

            latency_ms = int((time.monotonic() - start) * 1000)
            await queue.put(
                json.dumps(
                    {
                        "type": "thread_done",
                        "thread_id": thread_id,
                        "latency_ms": latency_ms,
                        "content": full_text,
                    }
                )
            )
        except Exception as exc:
            await queue.put(
                json.dumps(
                    {
                        "type": "error",
                        "thread_id": thread_id,
                        "error": str(exc),
                    }
                )
            )

    tasks = [asyncio.create_task(_pump(t)) for t in threads]
    remaining = len(tasks)

    while remaining > 0:
        item = await queue.get()
        yield f"data: {item}\n\n"
        remaining = sum(1 for t in tasks if not t.done())

    while not queue.empty():
        item = queue.get_nowait()
        yield f"data: {item}\n\n"

    yield f'data: {json.dumps({"type": "all_done"})}\n\n'
