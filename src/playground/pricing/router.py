from fastapi import APIRouter, Depends

from playground.auth.deps import get_current_user
from playground.db.models import User
from playground.pricing.schemas import PricingCatalogOut
from playground.pricing.service import pricing_catalog

router = APIRouter(tags=["pricing"])


@router.get("/pricing/models", response_model=PricingCatalogOut)
async def list_model_pricing(
    _user: User = Depends(get_current_user),
) -> PricingCatalogOut:
    return await pricing_catalog()
