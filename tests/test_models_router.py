from __future__ import annotations

import httpx
from httpx import ASGITransport, AsyncClient

from playground.app import create_app
from playground.auth.deps import get_current_user
from playground.db.connection import Database
from playground.db.models import Base, LlmModel, User
from playground.db.repos.model_repo import ModelRepo
from playground.deps import get_db, get_runtime_client


class FakeRuntime:
    def __init__(self) -> None:
        self.created = None

    async def list_models(self):
        return {
            "default_provider": "openrouter",
            "openrouter": {
                "models": [
                    {
                        "id": 9,
                        "provider": "openrouter",
                        "model_id": "vendor/model",
                        "name": "Vendor Model",
                        "enabled": True,
                        "supports_reasoning": True,
                        "sort_order": 3,
                        "config": {"tier": "test"},
                    }
                ]
            },
        }

    async def create_model(self, **kwargs):
        self.created = kwargs
        return {
            "id": 10,
            "provider": "openrouter",
            "model_id": "vendor/new",
            "name": "Vendor New",
            "enabled": True,
            "supports_reasoning": False,
            "sort_order": 80,
            "config": {"tier": "test"},
        }


async def make_db() -> Database:
    db = Database("sqlite+aiosqlite:///:memory:")
    db.connect()
    assert db.engine is not None
    async with db.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return db


async def add_model(db: Database) -> None:
    async with db.session() as session:
        session.add(
            LlmModel(
                provider="openrouter",
                model_name="vendor/model",
                display_name="Vendor Model",
                is_active=True,
                supports_reasoning=True,
                sort_order=3,
                config_json={"tier": "test"},
            )
        )


async def get_model(db: Database, provider: str, model_name: str) -> LlmModel | None:
    async with db.session() as session:
        return await ModelRepo(session).get_by_provider_model(provider, model_name)


async def test_models_response_includes_runtime_metadata() -> None:
    db = await make_db()
    app = create_app()
    app.dependency_overrides[get_db] = lambda: db

    try:
        await add_model(db)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/models")
    finally:
        await db.disconnect()

    assert resp.status_code == 200
    model = resp.json()["models"][0]
    assert model["supports_reasoning"] is True
    assert model["sort_order"] == 3
    assert model["config"] == {"tier": "test"}


async def test_models_sync_requires_auth_and_syncs_runtime_metadata() -> None:
    db = await make_db()
    app = create_app()
    app.dependency_overrides[get_db] = lambda: db
    runtime = FakeRuntime()
    app.dependency_overrides[get_runtime_client] = lambda: runtime

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            unauthorized = await client.post("/models/sync")

        assert unauthorized.status_code == 401

        app.dependency_overrides[get_current_user] = lambda: User(
            id=1,
            email="user@example.com",
            hashed_password="hashed",
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/models/sync")
        synced = await get_model(db, "openrouter", "vendor/model")
    finally:
        await db.disconnect()

    assert resp.status_code == 200
    assert resp.json() == {"synced": 1, "deactivated": 0}
    assert synced is not None
    assert synced.runtime_model_id == 9


async def test_create_model_requires_auth_and_proxies_runtime_model() -> None:
    db = await make_db()
    app = create_app()
    runtime = FakeRuntime()
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_runtime_client] = lambda: runtime

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            unauthorized = await client.post(
                "/models",
                json={"provider": "openrouter", "model_id": "vendor/new", "name": "Vendor New"},
            )

        assert unauthorized.status_code == 401

        app.dependency_overrides[get_current_user] = lambda: User(
            id=1,
            email="user@example.com",
            hashed_password="hashed",
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/models",
                json={
                    "provider": "openrouter",
                    "model_id": "vendor/new",
                    "name": "Vendor New",
                    "supports_reasoning": False,
                    "sort_order": 80,
                    "config": {"tier": "test"},
                },
            )
        synced = await get_model(db, "openrouter", "vendor/new")
    finally:
        await db.disconnect()

    assert resp.status_code == 200
    assert runtime.created == {
        "provider": "openrouter",
        "model_id": "vendor/new",
        "name": "Vendor New",
        "enabled": True,
        "supports_reasoning": False,
        "sort_order": 80,
        "config": {"tier": "test"},
    }
    assert synced is not None
    assert synced.runtime_model_id == 10
    assert resp.json()["model_name"] == "vendor/new"


async def test_create_model_surfaces_runtime_conflict() -> None:
    class ConflictRuntime(FakeRuntime):
        async def create_model(self, **kwargs):
            request = httpx.Request("POST", "http://runtime/models")
            response = httpx.Response(409, json={"detail": "Model already exists"})
            raise httpx.HTTPStatusError("conflict", request=request, response=response)

    db = await make_db()
    app = create_app()
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_runtime_client] = lambda: ConflictRuntime()
    app.dependency_overrides[get_current_user] = lambda: User(
        id=1,
        email="user@example.com",
        hashed_password="hashed",
    )

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/models",
                json={"provider": "openrouter", "model_id": "vendor/new", "name": "Vendor New"},
            )
    finally:
        await db.disconnect()

    assert resp.status_code == 409
    assert resp.json() == {"detail": "Model already exists"}
