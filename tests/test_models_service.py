from __future__ import annotations

from playground.db.connection import Database
from playground.db.models import Base
from playground.db.repos.model_repo import ModelRepo
from playground.models.service import sync_runtime_models


class FakeRuntime:
    async def list_models(self):
        return {
            "default_provider": "openrouter",
            "openrouter": {
                "default_model": "vendor/model",
                "models": [
                    {
                        "id": 42,
                        "provider": "openrouter",
                        "model_id": "vendor/model",
                        "name": "Vendor Model",
                        "enabled": True,
                        "supports_reasoning": True,
                        "sort_order": 7,
                        "config": {"tier": "test"},
                    }
                ],
            },
        }


async def test_sync_runtime_models_upserts_enriched_metadata() -> None:
    db = Database("sqlite+aiosqlite:///:memory:")
    db.connect()
    assert db.engine is not None
    async with db.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    try:
        repo = ModelRepo(db)
        synced = await sync_runtime_models(repo, FakeRuntime())
        models = await repo.list_active()
    finally:
        await db.disconnect()

    assert synced == 1
    assert len(models) == 1
    assert models[0].runtime_model_id == 42
    assert models[0].model_name == "vendor/model"
    assert models[0].display_name == "Vendor Model"
    assert models[0].supports_reasoning is True
    assert models[0].sort_order == 7
    assert models[0].config_json == {"tier": "test"}
