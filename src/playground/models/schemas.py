from typing import Any

from pydantic import BaseModel, Field


class ModelCreate(BaseModel):
    provider: str = Field(..., min_length=1, max_length=50)
    model_id: str = Field(..., min_length=1, max_length=200)
    name: str = Field(..., min_length=1, max_length=200)
    enabled: bool = True
    supports_reasoning: bool = False
    sort_order: int = 0
    config: dict[str, Any] | None = None


class ModelOut(BaseModel):
    id: str
    provider: str
    model_name: str
    display_name: str
    is_active: bool
    supports_reasoning: bool
    sort_order: int
    config: dict[str, Any] | None = None


class ModelsResponse(BaseModel):
    models: list[ModelOut]


class ModelsSyncResponse(BaseModel):
    synced: int
    deactivated: int
