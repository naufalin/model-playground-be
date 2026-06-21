"""Model registry endpoints."""

from fastapi import APIRouter, Depends

import playground.ids as ids
from playground.auth.deps import get_current_user
from playground.db.models import User
from playground.db.repos.model_repo import ModelRepo
from playground.deps import get_model_repo, get_runtime_client
from playground.models.schemas import ModelCreate, ModelOut, ModelsResponse, ModelsSyncResponse
from playground.models.service import create_runtime_model, sync_runtime_models
from playground.runtime.client import AgentRuntimeClient

router = APIRouter(tags=["models"])


def _model_out(model) -> ModelOut:  # noqa: ANN001
    return ModelOut(
        id=ids.encode(model.id),
        provider=model.provider,
        model_name=model.model_name,
        display_name=model.display_name,
        is_active=model.is_active,
        supports_reasoning=model.supports_reasoning,
        sort_order=model.sort_order,
        config=model.config_json,
    )


@router.get("/models", response_model=ModelsResponse)
async def list_models(
    provider: str | None = None,
    repo: ModelRepo = Depends(get_model_repo),
) -> ModelsResponse:
    models = await repo.list_active(provider=provider)
    return ModelsResponse(
        models=[_model_out(m) for m in models]
    )


@router.post("/models", response_model=ModelOut)
async def create_model(
    body: ModelCreate,
    _user: User = Depends(get_current_user),
    repo: ModelRepo = Depends(get_model_repo),
    runtime: AgentRuntimeClient = Depends(get_runtime_client),
) -> ModelOut:
    model = await create_runtime_model(body, repo, runtime)
    return _model_out(model)


@router.post("/models/sync")
async def sync_models(
    _user: User = Depends(get_current_user),
    repo: ModelRepo = Depends(get_model_repo),
    runtime: AgentRuntimeClient = Depends(get_runtime_client),
) -> ModelsSyncResponse:
    synced, deactivated = await sync_runtime_models(repo, runtime)
    return ModelsSyncResponse(synced=synced, deactivated=deactivated)
