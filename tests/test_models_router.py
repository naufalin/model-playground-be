from __future__ import annotations

from httpx import ASGITransport, AsyncClient

from playground.app import create_app
from playground.auth.deps import get_current_user
from playground.db.models import LlmModel, User
from playground.deps import get_model_repo, get_runtime_client


class FakeModelRepo:
    def __init__(self) -> None:
        self.upserted = []

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


class FakeRuntime:
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
    app.dependency_overrides[get_runtime_client] = lambda: FakeRuntime()

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
    assert resp.json() == {"synced": 1}
    assert repo.upserted[0]["runtime_model_id"] == 9
