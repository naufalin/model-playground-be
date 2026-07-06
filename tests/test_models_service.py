from __future__ import annotations

import httpx
import pytest
from fastapi import HTTPException

from playground.db.connection import Database
from playground.db.models import Base, LlmModel
from playground.db.repos.model_repo import ModelRepo
from playground.models.schemas import ModelCreate
from playground.models.service import create_runtime_model, sync_runtime_models


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


async def make_db() -> Database:
    db = Database("sqlite+aiosqlite:///:memory:")
    db.connect()
    assert db.engine is not None
    async with db.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return db


async def list_active_models(db: Database) -> list[LlmModel]:
    async with db.session() as session:
        return await ModelRepo(session).list_active()


async def get_model(
    db: Database,
    provider: str,
    model_name: str,
    *,
    active_only: bool = True,
) -> LlmModel | None:
    async with db.session() as session:
        return await ModelRepo(session).get_by_provider_model(
            provider,
            model_name,
            active_only=active_only,
        )


async def test_sync_runtime_models_upserts_enriched_metadata() -> None:
    db = await make_db()

    try:
        synced, deactivated = await sync_runtime_models(db, FakeRuntime())
        models = await list_active_models(db)
    finally:
        await db.disconnect()

    assert synced == 1
    assert deactivated == 0
    assert len(models) == 1
    assert models[0].runtime_model_id == 42
    assert models[0].model_name == "vendor/model"
    assert models[0].display_name == "Vendor Model"
    assert models[0].supports_reasoning is True
    assert models[0].sort_order == 7
    assert models[0].config_json == {"tier": "test"}


async def test_sync_runtime_models_deactivates_missing_local_models() -> None:
    db = await make_db()
    try:
        async with db.session() as session:
            session.add_all(
                [
                    LlmModel(
                        provider="openai",
                        model_name="legacy-model",
                        display_name="Legacy Model",
                        is_active=True,
                    ),
                    LlmModel(
                        provider="openrouter",
                        model_name="vendor/model",
                        display_name="Old Name",
                        is_active=True,
                    ),
                ]
            )

        synced, deactivated = await sync_runtime_models(db, FakeRuntime())
        active_models = await list_active_models(db)
        legacy = await get_model(db, "openai", "legacy-model", active_only=False)
    finally:
        await db.disconnect()

    assert synced == 1
    assert deactivated == 1
    assert [(model.provider, model.model_name) for model in active_models] == [
        ("openrouter", "vendor/model")
    ]
    assert legacy is not None
    assert legacy.is_active is False


async def test_sync_runtime_models_hides_runtime_disabled_models() -> None:
    class DisabledRuntime(FakeRuntime):
        async def list_models(self):
            payload = await super().list_models()
            payload["openrouter"]["models"][0]["enabled"] = False
            return payload

    db = await make_db()
    try:
        synced, deactivated = await sync_runtime_models(db, DisabledRuntime())
        models = await list_active_models(db)
        disabled = await get_model(db, "openrouter", "vendor/model", active_only=False)
    finally:
        await db.disconnect()

    assert synced == 1
    assert deactivated == 0
    assert models == []
    assert disabled is not None
    assert disabled.is_active is False


async def test_create_runtime_model_proxies_runtime_and_upserts_local_model() -> None:
    class CreateRuntime:
        def __init__(self) -> None:
            self.payload = None

        async def create_model(self, **kwargs):
            self.payload = kwargs
            return {
                "id": 99,
                "provider": "openrouter",
                "model_id": "vendor/new",
                "name": "Vendor New",
                "enabled": True,
                "supports_reasoning": False,
                "sort_order": 80,
                "config": {"tier": "test"},
            }

    db = await make_db()
    runtime = CreateRuntime()
    try:
        model = await create_runtime_model(
            ModelCreate(
                provider="openrouter",
                model_id="vendor/new",
                name="Vendor New",
                supports_reasoning=False,
                sort_order=80,
                config={"tier": "test"},
            ),
            db,
            runtime,
        )
    finally:
        await db.disconnect()

    assert runtime.payload == {
        "provider": "openrouter",
        "model_id": "vendor/new",
        "name": "Vendor New",
        "enabled": True,
        "supports_reasoning": False,
        "sort_order": 80,
        "config": {"tier": "test"},
    }
    assert model.runtime_model_id == 99
    assert model.model_name == "vendor/new"
    assert model.config_json == {"tier": "test"}


@pytest.mark.parametrize("runtime_status, expected_status", [(400, 400), (409, 409), (500, 502)])
async def test_create_runtime_model_maps_runtime_errors(
    runtime_status: int,
    expected_status: int,
) -> None:
    class ErrorRuntime:
        async def create_model(self, **kwargs):
            request = httpx.Request("POST", "http://runtime/models")
            response = httpx.Response(runtime_status, json={"detail": "runtime failed"})
            raise httpx.HTTPStatusError("error", request=request, response=response)

    db = await make_db()
    try:
        with pytest.raises(HTTPException) as exc:
            await create_runtime_model(
                ModelCreate(provider="openrouter", model_id="vendor/new", name="Vendor New"),
                db,
                ErrorRuntime(),
            )
    finally:
        await db.disconnect()

    assert exc.value.status_code == expected_status
    assert exc.value.detail == "runtime failed"


async def test_create_runtime_model_maps_runtime_request_errors() -> None:
    class ErrorRuntime:
        async def create_model(self, **kwargs):
            request = httpx.Request("POST", "http://runtime/models")
            raise httpx.ConnectError("unreachable", request=request)

    db = await make_db()
    try:
        with pytest.raises(HTTPException) as exc:
            await create_runtime_model(
                ModelCreate(provider="openrouter", model_id="vendor/new", name="Vendor New"),
                db,
                ErrorRuntime(),
            )
    finally:
        await db.disconnect()

    assert exc.value.status_code == 502
    assert exc.value.detail == "Could not reach agent runtime"
