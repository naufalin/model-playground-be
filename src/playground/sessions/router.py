"""Playground CRUD and chat streaming endpoints."""

from typing import NoReturn

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from playground.auth.deps import get_current_user
from playground.db.models import User
from playground.deps import get_db, get_runtime_client
from playground.sessions.schemas import (
    ChatRequest,
    ContinueChatRequest,
    PlaygroundCreate,
    PlaygroundDetail,
    PlaygroundListOut,
    PlaygroundOut,
)
from playground.sessions.service import PlaygroundError, PlaygroundService

router = APIRouter(prefix="/playground", tags=["playground"])

STREAM_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


async def get_playground_service(
    db=Depends(get_db),
    runtime=Depends(get_runtime_client),
) -> PlaygroundService:
    return PlaygroundService(db, runtime)


def _raise_http_error(exc: PlaygroundError) -> NoReturn:
    raise HTTPException(status_code=exc.status_code, detail=exc.detail)


@router.post("", status_code=status.HTTP_201_CREATED, response_model=PlaygroundOut)
async def create_playground(
    body: PlaygroundCreate,
    user: User = Depends(get_current_user),
    service: PlaygroundService = Depends(get_playground_service),
) -> PlaygroundOut:
    return await service.create_playground(user_id=user.id, title=body.title)


@router.get("", response_model=PlaygroundListOut)
async def list_playgrounds(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user: User = Depends(get_current_user),
    service: PlaygroundService = Depends(get_playground_service),
) -> PlaygroundListOut:
    return await service.list_playgrounds(user_id=user.id, limit=limit, offset=offset)


@router.get("/{encoded_id}", response_model=PlaygroundDetail)
async def get_playground(
    encoded_id: str,
    user: User = Depends(get_current_user),
    service: PlaygroundService = Depends(get_playground_service),
) -> PlaygroundDetail:
    try:
        return await service.get_playground(encoded_id, user.id)
    except PlaygroundError as exc:
        _raise_http_error(exc)


@router.delete("/{encoded_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_playground(
    encoded_id: str,
    user: User = Depends(get_current_user),
    service: PlaygroundService = Depends(get_playground_service),
) -> None:
    try:
        await service.delete_playground(encoded_id, user.id)
    except PlaygroundError as exc:
        _raise_http_error(exc)


@router.post("/{encoded_id}/chat")
async def chat_multi(
    encoded_id: str,
    body: ChatRequest,
    user: User = Depends(get_current_user),
    service: PlaygroundService = Depends(get_playground_service),
) -> StreamingResponse:
    try:
        stream = await service.stream_multi_chat(
            encoded_id,
            user.id,
            body.message,
            [(model.provider, model.model_name) for model in body.models],
        )
    except PlaygroundError as exc:
        _raise_http_error(exc)

    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers=STREAM_HEADERS,
    )


@router.post("/{encoded_id}/chat/{thread_encoded_id}")
async def chat_single(
    encoded_id: str,
    thread_encoded_id: str,
    body: ContinueChatRequest,
    user: User = Depends(get_current_user),
    service: PlaygroundService = Depends(get_playground_service),
) -> StreamingResponse:
    try:
        stream = await service.stream_single_chat(
            encoded_id,
            thread_encoded_id,
            user.id,
            body.message,
        )
    except PlaygroundError as exc:
        _raise_http_error(exc)

    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers=STREAM_HEADERS,
    )
