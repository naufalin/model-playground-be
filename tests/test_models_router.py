from __future__ import annotations

import httpx
from httpx import ASGITransport, AsyncClient

from playground.app import create_app
from playground.auth.deps import get_current_user
from playground.db.models import LlmModel, User
from playground.deps import get_model_repo, get_runtime_client


class FakeModelRepo:
    def __init__(self) -> None:
        self.upserted = []
        self.deactivated_seen = None

    async def list_active(self, provider: str | None = None):
        return [
            LlmModel(
                id=1,
                provider=provider or "openrouter",
                model_name="vendor/model",
                display_name="Vendor Model",
                is_active=True,
                supports_reasoning=True,
                sort_order=3,
                config_json={"tier": "test"},
            )
        ]

    async def upsert_runtime_model(self, **kwargs):
        self.upserted.append(kwargs)
        return LlmModel(
            id=2,
            provider=kwargs["provider"],
            model_name=kwargs["model_name"],
            display_name=kwargs["display_name"],
            is_active=kwargs["is_active"],
            supports_reasoning=kwargs["supports_reasoning"],
            sort_order=kwargs["sort_order"],
            config_json=kwargs["config_json"],
        )

    async def deactivate_missing_runtime_models(self, seen_models):
        self.deactivated_seen = seen_models
        return 0


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


async def test_models_response_includes_runtime_metadata() -> None:
    app = create_app()
    repo = FakeModelRepo()
    app.dependency_overrides[get_model_repo] = lambda: repo

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/models")

    assert resp.status_code == 200
    model = resp.json()["models"][0]
    assert model["supports_reasoning"] is True
    assert model["sort_order"] == 3
    assert model["config"] == {"tier": "test"}


async def test_models_sync_requires_auth_and_syncs_runtime_metadata() -> None:
    app = create_app()
    repo = FakeModelRepo()
    app.dependency_overrides[get_model_repo] = lambda: repo
    runtime = FakeRuntime()
    app.dependency_overrides[get_runtime_client] = lambda: runtime

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

    assert resp.status_code == 200
    assert resp.json() == {"synced": 1, "deactivated": 0}
    assert repo.upserted[0]["runtime_model_id"] == 9


async def test_create_model_requires_auth_and_proxies_runtime_model() -> None:
    app = create_app()
    repo = FakeModelRepo()
    runtime = FakeRuntime()
    app.dependency_overrides[get_model_repo] = lambda: repo
    app.dependency_overrides[get_runtime_client] = lambda: runtime

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
    assert repo.upserted[0]["runtime_model_id"] == 10
    assert resp.json()["model_name"] == "vendor/new"


async def test_create_model_surfaces_runtime_conflict() -> None:
    class ConflictRuntime(FakeRuntime):
        async def create_model(self, **kwargs):
            request = httpx.Request("POST", "http://runtime/models")
            response = httpx.Response(409, json={"detail": "Model already exists"})
            raise httpx.HTTPStatusError("conflict", request=request, response=response)

    app = create_app()
    repo = FakeModelRepo()
    app.dependency_overrides[get_model_repo] = lambda: repo
    app.dependency_overrides[get_runtime_client] = lambda: ConflictRuntime()
    app.dependency_overrides[get_current_user] = lambda: User(
        id=1,
        email="user@example.com",
        hashed_password="hashed",
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/models",
            json={"provider": "openrouter", "model_id": "vendor/new", "name": "Vendor New"},
        )

    assert resp.status_code == 409
    assert resp.json() == {"detail": "Model already exists"}
