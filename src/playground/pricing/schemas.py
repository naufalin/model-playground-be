from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class ModelPriceOut(BaseModel):
    provider: str
    model_name: str
    display_name: str
    input_per_million_usd: float | None = None
    cached_input_per_million_usd: float | None = None
    output_per_million_usd: float | None = None
    rate_kind: Literal["exact", "from", "unavailable"]
    source_url: str
    refreshed_at: datetime


class PricingCatalogOut(BaseModel):
    models: list[ModelPriceOut]
    refreshed_at: datetime
