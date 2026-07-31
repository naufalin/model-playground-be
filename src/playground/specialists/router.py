"""Authenticated proxy for the runtime specialist catalog."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Response, status

from playground.auth.deps import get_current_user
from playground.db.models import User
from playground.deps import get_runtime_client
from playground.runtime.client import AgentRuntimeClient

router = APIRouter(prefix="/specialists", tags=["specialists"])
CurrentUser = Annotated[User, Depends(get_current_user)]
RuntimeClient = Annotated[AgentRuntimeClient, Depends(get_runtime_client)]


@router.get("")
async def list_specialists(_: CurrentUser, runtime: RuntimeClient):
    return await runtime.list_specialists()


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_specialist(payload: dict[str, Any], _: CurrentUser, runtime: RuntimeClient):
    return await runtime.create_specialist(payload)


@router.patch("/{name}")
async def update_specialist(
    name: str,
    payload: dict[str, Any],
    _: CurrentUser,
    runtime: RuntimeClient,
):
    return await runtime.update_specialist(name, payload)


@router.delete("/{name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_specialist(name: str, _: CurrentUser, runtime: RuntimeClient):
    await runtime.delete_specialist(name)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
