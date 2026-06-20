"""GET /models — list active LLM models."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

import playground.ids as ids
from playground.db.repos.model_repo import ModelRepo
from playground.deps import get_model_repo

router = APIRouter(tags=["models"])


class ModelOut(BaseModel):
    id: str
    provider: str
    model_name: str
    display_name: str


class ModelsResponse(BaseModel):
    models: list[ModelOut]


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
            )
            for m in models
        ]
    )
