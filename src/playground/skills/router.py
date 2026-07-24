"""Read-only proxy for the runtime skill catalog."""

from fastapi import APIRouter, Depends

from playground.auth.deps import get_current_user
from playground.db.models import User
from playground.deps import get_runtime_client
from playground.runtime.client import AgentRuntimeClient

router = APIRouter(prefix="/skills", tags=["skills"])


@router.get("")
async def list_skills(
    _user: User = Depends(get_current_user),
    runtime: AgentRuntimeClient = Depends(get_runtime_client),
) -> dict:
    return await runtime.list_skills()
