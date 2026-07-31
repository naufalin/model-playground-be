"""Read-only proxy for reusable runtime system prompts."""

from typing import Any

from fastapi import APIRouter, Depends

from playground.auth.deps import get_current_user
from playground.db.models import User
from playground.deps import get_runtime_client
from playground.runtime.client import AgentRuntimeClient

router = APIRouter(prefix="/prompts", tags=["prompts"])


@router.get("")
async def list_prompts(
    _user: User = Depends(get_current_user),  # noqa: B008
    runtime: AgentRuntimeClient = Depends(get_runtime_client),  # noqa: B008
) -> dict[str, Any]:
    return await runtime.list_prompts()
