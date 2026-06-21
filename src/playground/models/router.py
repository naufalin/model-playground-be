"""Model registry endpoints."""

from fastapi import APIRouter, Depends

import playground.ids as ids
from playground.auth.deps import get_current_user
from playground.db.models import User
from playground.db.repos.model_repo import ModelRepo
from playground.deps import get_model_repo, get_runtime_client
from playground.models.schemas import ModelOut, ModelsResponse
from playground.models.service import sync_runtime_models
from playground.runtime.client import AgentRuntimeClient

router = APIRouter(tags=["models"])


@router.get("/models", response_model=ModelsResponse)
async def list_models(
    provider: str | None = None,
    repo: ModelRepo = Depends(get_model_repo),
) -> ModelsResponse:
    models = await repo.list_active(provider=provider)
    return ModelsResponse(
        models=[
            ModelOut(
                id=ids.encode(m.id),
                provider=m.provider,
                model_name=m.model_name,
                display_name=m.display_name,
                is_active=m.is_active,
                supports_reasoning=m.supports_reasoning,
                sort_order=m.sort_order,
                config=m.config_json,
            )
            for m in models
        ]
    )


@router.post("/models/sync")
async def sync_models(
    _user: User = Depends(get_current_user),
    repo: ModelRepo = Depends(get_model_repo),
    runtime: AgentRuntimeClient = Depends(get_runtime_client),
) -> dict[str, int]:
    synced = await sync_runtime_models(repo, runtime)
    return {"synced": synced}
