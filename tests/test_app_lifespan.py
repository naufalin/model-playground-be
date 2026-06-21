from __future__ import annotations

from playground.app import create_app, lifespan
from playground.config import Settings
from playground.db.repos.model_repo import ModelRepo
from playground.runtime.client import AgentRuntimeClient


async def test_lifespan_initializes_and_closes_app_resources(monkeypatch) -> None:
    async def list_active(self):
        return []

    async def sync_runtime_models(repo, runtime):
        return 0

    monkeypatch.setattr(ModelRepo, "list_active", list_active)
    monkeypatch.setattr("playground.app.sync_runtime_models", sync_runtime_models)
    app = create_app(
        Settings(
            secret_key="test",
            database_url="sqlite+aiosqlite:///:memory:",
            cors_origins=["http://example.com"],
        )
    )

    async with lifespan(app):
        assert app.state.db.engine is not None
        assert isinstance(app.state.runtime_client, AgentRuntimeClient)

    assert app.state.db.engine is None
