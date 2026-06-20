"""SQLAlchemy async engine and session factory."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine


class Database:
    """Wraps a SQLAlchemy async engine."""

    def __init__(self, url: str):
        self.url = url
        self.engine: AsyncEngine | None = None

    def connect(self) -> None:
        """Create the async engine (no actual connection until first query)."""
        url = self.url
        # Auto-fix: ensure async driver prefix for PostgreSQL
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        engine_kwargs = {}
        if url.startswith("postgresql+asyncpg://"):
            engine_kwargs = {"pool_size": 10, "max_overflow": 20}
        self.engine = create_async_engine(url, **engine_kwargs)

    async def disconnect(self) -> None:
        """Dispose of the engine and all connections."""
        if self.engine:
            await self.engine.dispose()
            self.engine = None

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession]:
        """Yield an AsyncSession that auto-commits on success, rolls back on error."""
        assert self.engine is not None, "Database not connected — call connect() first"
        async with AsyncSession(self.engine, expire_on_commit=False) as session, session.begin():
            yield session
