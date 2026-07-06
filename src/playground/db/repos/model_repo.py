"""Model registry repository."""

from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from playground.db.models import LlmModel


class ModelRepo:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def list_active(self, provider: str | None = None) -> list[LlmModel]:
        """Return active models, optionally filtered by provider."""
        stmt = select(LlmModel).where(LlmModel.is_active == True)  # noqa: E712
        if provider:
            stmt = stmt.where(LlmModel.provider == provider)
        stmt = stmt.order_by(LlmModel.provider, LlmModel.sort_order, LlmModel.display_name)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def list_all(self) -> list[LlmModel]:
        result = await self.session.execute(
            select(LlmModel).order_by(
                LlmModel.provider,
                LlmModel.sort_order,
                LlmModel.display_name,
            )
        )
        return list(result.scalars().all())

    async def get_by_provider_model(
        self,
        provider: str,
        model_name: str,
        *,
        active_only: bool = True,
    ) -> LlmModel | None:
        """Look up a model by its provider + model_name pair."""
        stmt = select(LlmModel).where(
            LlmModel.provider == provider,
            LlmModel.model_name == model_name,
        )
        if active_only:
            stmt = stmt.where(LlmModel.is_active.is_(True))
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_id(self, model_id: int) -> LlmModel | None:
        return await self.session.get(LlmModel, model_id)

    async def set_active(self, model_id: int, is_active: bool) -> None:
        """Admin toggle: enable/disable a model without deleting."""
        model = await self.session.get(LlmModel, model_id)
        if model:
            model.is_active = is_active

    async def upsert_runtime_model(
        self,
        *,
        runtime_model_id: int | None,
        provider: str,
        model_name: str,
        display_name: str,
        is_active: bool,
        supports_reasoning: bool,
        sort_order: int,
        config_json: dict[str, Any] | None,
    ) -> LlmModel:
        result = await self.session.execute(
            select(LlmModel).where(
                LlmModel.provider == provider,
                LlmModel.model_name == model_name,
            )
        )
        model = result.scalar_one_or_none()
        if model is None:
            model = LlmModel(
                provider=provider,
                model_name=model_name,
                display_name=display_name,
            )
            self.session.add(model)

        model.runtime_model_id = runtime_model_id
        model.display_name = display_name
        model.is_active = is_active
        model.supports_reasoning = supports_reasoning
        model.sort_order = sort_order
        model.config_json = config_json
        await self.session.flush()
        return model

    async def deactivate_missing_runtime_models(
        self,
        seen_models: set[tuple[str, str]],
    ) -> int:
        """Deactivate local models that are no longer present in the runtime registry."""
        active_models = await self.session.execute(
            select(LlmModel).where(LlmModel.is_active.is_(True))
        )
        missing_ids = [
            model.id
            for model in active_models.scalars()
            if (model.provider, model.model_name) not in seen_models
        ]
        if not missing_ids:
            return 0
        result = await self.session.execute(
            update(LlmModel)
            .where(LlmModel.id.in_(missing_ids))
            .values(is_active=False)
        )
        await self.session.flush()
        return result.rowcount or 0
