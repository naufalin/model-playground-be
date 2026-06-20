"""Model registry repository."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from playground.db.connection import Database
from playground.db.models import LlmModel


class ModelRepo:
    def __init__(self, db: Database | AsyncSession):
        self.db = db

    @asynccontextmanager
    async def _session(self) -> AsyncGenerator[AsyncSession]:
        if isinstance(self.db, AsyncSession):
            yield self.db
        else:
            async with self.db.session() as session:
                yield session

    async def list_active(self, provider: str | None = None) -> list[LlmModel]:
        """Return active models, optionally filtered by provider."""
        async with self._session() as s:
            stmt = select(LlmModel).where(LlmModel.is_active == True)  # noqa: E712
            if provider:
                stmt = stmt.where(LlmModel.provider == provider)
            stmt = stmt.order_by(LlmModel.provider, LlmModel.display_name)
            result = await s.execute(stmt)
            return list(result.scalars().all())

    async def get_by_provider_model(self, provider: str, model_name: str) -> LlmModel | None:
        """Look up a model by its provider + model_name pair."""
        async with self._session() as s:
            result = await s.execute(
                select(LlmModel).where(
                    LlmModel.provider == provider,
                    LlmModel.model_name == model_name,
                )
            )
            return result.scalar_one_or_none()

    async def get_by_id(self, model_id: int) -> LlmModel | None:
        async with self._session() as s:
            return await s.get(LlmModel, model_id)

    async def set_active(self, model_id: int, is_active: bool) -> None:
        """Admin toggle: enable/disable a model without deleting."""
        async with self._session() as s:
            model = await s.get(LlmModel, model_id)
            if model:
                model.is_active = is_active
