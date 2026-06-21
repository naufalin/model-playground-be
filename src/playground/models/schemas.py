from typing import Any

from pydantic import BaseModel


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
