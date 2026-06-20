"""Playground CRUD and chat streaming endpoints."""

from __future__ import annotations

import json
import time

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from playground.auth.deps import get_current_user
from playground.db.models import User
from playground.db.repos.model_repo import ModelRepo
from playground.db.repos.session_repo import SessionRepo
from playground.db.repos.thread_repo import ThreadRepo
from playground.deps import get_model_repo, get_session_repo, get_thread_repo
from playground.ids import decode, encode
from playground.playground.fanout import fanout_chat
from playground.playground.schemas import (
    ChatRequest,
    ContinueChatRequest,
    MessageOut,
    PlaygroundCreate,
    PlaygroundDetail,
    PlaygroundListOut,
    PlaygroundOut,
    ThreadOut,
)
from playground.runtime.client import AgentRuntimeClient

router = APIRouter(prefix="/playground", tags=["playground"])

# ---------------------------------------------------------------------------
# Lazy singleton for the agent runtime client
# ---------------------------------------------------------------------------
_runtime_client: AgentRuntimeClient | None = None


def _get_runtime() -> AgentRuntimeClient:
    global _runtime_client
    if _runtime_client is None:
        from playground.config import settings

        _runtime_client = AgentRuntimeClient(base_url=settings.agent_runtime_url)
    return _runtime_client


# ---------------------------------------------------------------------------
# CRUD endpoints
# ---------------------------------------------------------------------------

@router.post("", status_code=status.HTTP_201_CREATED, response_model=PlaygroundOut)
async def create_playground(
    body: PlaygroundCreate,
    user: User = Depends(get_current_user),
    session_repo: SessionRepo = Depends(get_session_repo),
):
    session = await session_repo.create(user_id=user.id, title=body.title)
    return PlaygroundOut(
        id=encode(session.id), title=session.title, created_at=session.created_at
    )


@router.get("", response_model=PlaygroundListOut)
async def list_playgrounds(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user: User = Depends(get_current_user),
    session_repo: SessionRepo = Depends(get_session_repo),
):
    sessions = await session_repo.list_by_user(user_id=user.id, limit=limit, offset=offset)
    # TODO: add a count query in SessionRepo for accurate total with pagination
    total = len(sessions)
    items = [
        PlaygroundOut(id=encode(s.id), title=s.title, created_at=s.created_at)
        for s in sessions
    ]
    return PlaygroundListOut(sessions=items, total=total)


@router.get("/{encoded_id}", response_model=PlaygroundDetail)
async def get_playground(
    encoded_id: str,
    user: User = Depends(get_current_user),
    session_repo: SessionRepo = Depends(get_session_repo),
    thread_repo: ThreadRepo = Depends(get_thread_repo),
    model_repo: ModelRepo = Depends(get_model_repo),
):
    session_id = decode(encoded_id)
    session = await session_repo.get(session_id)
    if session is None or session.user_id != user.id:
        raise HTTPException(status_code=404, detail="Playground not found")

    threads = await thread_repo.get_by_session(session_id)
    thread_outs: list[ThreadOut] = []
    for t in threads:
        model = await model_repo.get_by_provider_model(t.provider, t.model_name)
        display_name = model.display_name if model else t.model_name
        messages = [
            MessageOut(
                id=m.id,
                role=m.role,
                content=m.content,
                latency_ms=m.latency_ms,
                created_at=m.created_at,
            )
            for m in await thread_repo.get_messages(t.id)
        ]
        thread_outs.append(
            ThreadOut(
                id=encode(t.id),
                provider=t.provider,
                model_name=t.model_name,
                display_name=display_name,
                messages=messages,
            )
        )

    return PlaygroundDetail(
        id=encode(session.id),
        title=session.title,
        created_at=session.created_at,
        threads=thread_outs,
    )


@router.delete("/{encoded_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_playground(
    encoded_id: str,
    user: User = Depends(get_current_user),
    session_repo: SessionRepo = Depends(get_session_repo),
):
    session_id = decode(encoded_id)
    session = await session_repo.get_if_owner(session_id, user.id)
    if session is None:
        raise HTTPException(status_code=404, detail="Playground not found")
    await session_repo.delete(session_id)


# ---------------------------------------------------------------------------
# Chat endpoints
# ---------------------------------------------------------------------------

STREAM_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


@router.post("/{encoded_id}/chat")
async def chat_multi(
    encoded_id: str,
    body: ChatRequest,
    user: User = Depends(get_current_user),
    session_repo: SessionRepo = Depends(get_session_repo),
    thread_repo: ThreadRepo = Depends(get_thread_repo),
    model_repo: ModelRepo = Depends(get_model_repo),
):
    session_id = decode(encoded_id)
    session = await session_repo.get(session_id)
    if session is None or session.user_id != user.id:
        raise HTTPException(status_code=404, detail="Playground not found")

    runtime = _get_runtime()

    # Validate models and find/create threads
    threads = []
    for sel in body.models:
        model = await model_repo.get_by_provider_model(sel.provider, sel.model_name)
        if model is None:
            raise HTTPException(
                status_code=400,
                detail=f"Model not found: {sel.provider}/{sel.model_name}",
            )
        thread = await thread_repo.get_by_session_and_model(
            session_id, sel.provider, sel.model_name
        )
        if thread is None:
            runtime_session_id = await runtime.create_session(sel.provider, sel.model_name)
            thread = await thread_repo.create(
                playground_session_id=session_id,
                provider=sel.provider,
                model_name=sel.model_name,
                runtime_session_id=runtime_session_id,
                model_id=model.id,
            )
        threads.append(thread)

    # Save user message to each thread
    for t in threads:
        await thread_repo.add_message(t.id, role="user", content=body.message)

    # Collect assistant text per thread for persistence
    thread_texts: dict[int, str] = {t.id: "" for t in threads}
    thread_latencies: dict[int, int] = {}

    async def _stream():
        async for chunk in fanout_chat(runtime, threads, body.message):
            # Parse chunk to collect text for persistence
            try:
                data = json.loads(chunk.removeprefix("data: ").strip())
                tid_encoded = data.get("thread_id")
                if tid_encoded and data.get("type") == "text_delta":
                    tid = decode(tid_encoded)
                    thread_texts[tid] += data.get("delta", "")
                if tid_encoded and data.get("type") == "thread_done":
                    tid = decode(tid_encoded)
                    thread_latencies[tid] = data.get("latency_ms", 0)
            except (json.JSONDecodeError, ValueError):
                pass
            yield chunk

        # Save assistant responses after stream completes
        for t in threads:
            content = thread_texts.get(t.id, "")
            if content:
                await thread_repo.add_message(
                    t.id,
                    role="assistant",
                    content=content,
                    latency_ms=thread_latencies.get(t.id),
                )

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers=STREAM_HEADERS,
    )


@router.post("/{encoded_id}/chat/{thread_encoded_id}")
async def chat_single(
    encoded_id: str,
    thread_encoded_id: str,
    body: ContinueChatRequest,
    user: User = Depends(get_current_user),
    session_repo: SessionRepo = Depends(get_session_repo),
    thread_repo: ThreadRepo = Depends(get_thread_repo),
):
    session_id = decode(encoded_id)
    session = await session_repo.get(session_id)
    if session is None or session.user_id != user.id:
        raise HTTPException(status_code=404, detail="Playground not found")

    thread_id = decode(thread_encoded_id)
    thread = await thread_repo.get(thread_id)
    if thread is None or thread.playground_session_id != session_id:
        raise HTTPException(status_code=404, detail="Thread not found")

    runtime = _get_runtime()

    # Save user message
    await thread_repo.add_message(thread_id, role="user", content=body.message)

    thread_id_enc = encode(thread_id)
    full_text = ""
    latency_ms = 0

    async def _stream():
        nonlocal full_text, latency_ms
        start = time.monotonic()
        try:
            async for event in runtime.chat_stream(thread.runtime_session_id, body.message):
                event["thread_id"] = thread_id_enc
                if event.get("type") == "text_delta":
                    full_text += event.get("delta", "")
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:
            error_event = {
                "type": "error",
                "thread_id": thread_id_enc,
                "error": str(exc),
            }
            yield f"data: {json.dumps(error_event)}\n\n"
        latency_ms = int((time.monotonic() - start) * 1000)

        # Save assistant response
        if full_text:
            await thread_repo.add_message(
                thread_id, role="assistant", content=full_text, latency_ms=latency_ms
            )

        done_event = {
            "type": "thread_done",
            "thread_id": thread_id_enc,
            "latency_ms": latency_ms,
            "content": full_text,
        }
        yield f"data: {json.dumps(done_event)}\n\n"
        yield f"data: {json.dumps({'type': 'all_done'})}\n\n"

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers=STREAM_HEADERS,
    )
