"""Tool registry proxy endpoints."""

from typing import Any

from fastapi import APIRouter, Depends

from playground.auth.deps import get_current_user
from playground.db.models import User
from playground.deps import get_runtime_client
from playground.runtime.client import AgentRuntimeClient

router = APIRouter(prefix="/tools", tags=["tools"])


@router.get("")
async def list_tools(
    _user: User = Depends(get_current_user),
    runtime: AgentRuntimeClient = Depends(get_runtime_client),
) -> dict[str, Any]:
    return await runtime.list_tools()
