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
    threads: list[tuple[ModelThread, str | None]],
    user_message: str,
) -> AsyncGenerator[str, None]:
    """Merge N per-thread streaming responses into a single SSE stream."""
    queue: asyncio.Queue[str] = asyncio.Queue()

    async def _pump(thread: ModelThread, reasoning_effort: str | None) -> None:
        thread_id = encode(thread.id)
        try:
            await queue.put(json.dumps({"type": "thread_start", "thread_id": thread_id}))

            full_text = ""
            start = time.monotonic()
            done_event: dict | None = None
            async for event in runtime.chat_stream(
                thread.runtime_session_id,
                user_message,
                provider=thread.provider,
                model=thread.model_name,
                reasoning_effort=reasoning_effort,
            ):
                if event.get("type") == "done":
                    done_event = event
                    continue
                event["thread_id"] = thread_id
                await queue.put(json.dumps(event))
                if event.get("type") == "text_delta":
                    full_text += event.get("delta", "")

            latency_ms = int((time.monotonic() - start) * 1000)
            content = full_text
            if done_event:
                content = done_event.get("content") or full_text
            await queue.put(
                json.dumps(
                    {
                        "type": "thread_done",
                        "thread_id": thread_id,
                        "latency_ms": latency_ms,
                        "content": content,
                        "provider": (done_event or {}).get("provider") or thread.provider,
                        "model": (done_event or {}).get("model") or thread.model_name,
                        "usage": (done_event or {}).get("usage"),
                        "thinking": (done_event or {}).get("thinking"),
                        "output_delta_count": (done_event or {}).get("output_delta_count"),
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

    tasks = [
        asyncio.create_task(_pump(thread, reasoning_effort))
        for thread, reasoning_effort in threads
    ]
    remaining = len(tasks)

    while remaining > 0:
        item = await queue.get()
        yield f"data: {item}\n\n"
        remaining = sum(1 for t in tasks if not t.done())

    while not queue.empty():
        item = queue.get_nowait()
        yield f"data: {item}\n\n"

    yield f'data: {json.dumps({"type": "all_done"})}\n\n'
