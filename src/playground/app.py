"""FastAPI application with lifespan management."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from playground.auth.router import router as auth_router
from playground.config import Settings, get_settings
from playground.db.connection import Database
from playground.db.repos.model_repo import ModelRepo
from playground.models.router import router as models_router
from playground.runtime.client import AgentRuntimeClient
from playground.sessions.router import router as playground_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings

    db = Database(settings.database_url)
    db.connect()
    app.state.db = db

    runtime_client = AgentRuntimeClient(base_url=settings.agent_runtime_url)
    app.state.runtime_client = runtime_client

    model_repo = ModelRepo(db)
    models = await model_repo.list_active()
    if not models:
        import logging

        logging.warning("No active models in registry — run alembic upgrade head first")
    yield

    await runtime_client.close()
    await db.disconnect()


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(
        title="LLM Playground",
        description="Test multiple LLMs side-by-side",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = settings or get_settings()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=app.state.settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(auth_router)
    app.include_router(models_router)
    app.include_router(playground_router)

    @app.get("/")
    async def health():
        return {"status": "ok", "version": "0.1.0"}

    return app


app = create_app()
