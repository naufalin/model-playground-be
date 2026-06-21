from __future__ import annotations

from typing import Any

from playground.db.repos.model_repo import ModelRepo
from playground.runtime.client import AgentRuntimeClient


def _iter_runtime_models(payload: dict[str, Any]):
    for provider, provider_payload in payload.items():
        if provider == "default_provider" or not isinstance(provider_payload, dict):
            continue
        models = provider_payload.get("models")
        if not isinstance(models, list):
            continue
        for model in models:
            if isinstance(model, dict):
                yield provider, model


async def sync_runtime_models(repo: ModelRepo, runtime: AgentRuntimeClient) -> int:
    """Sync model metadata from agent runtime into the local registry."""
    payload = await runtime.list_models()
    synced = 0
    for provider_key, model in _iter_runtime_models(payload):
        provider = str(model.get("provider") or provider_key).lower()
        model_name = model.get("model_id")
        if not isinstance(model_name, str) or not model_name:
            continue
        await repo.upsert_runtime_model(
            runtime_model_id=model.get("id") if isinstance(model.get("id"), int) else None,
            provider=provider,
            model_name=model_name,
            display_name=str(model.get("name") or model_name),
            is_active=bool(model.get("enabled", True)),
            supports_reasoning=bool(model.get("supports_reasoning", False)),
            sort_order=int(model.get("sort_order") or 0),
            config_json=model.get("config") if isinstance(model.get("config"), dict) else None,
        )
        synced += 1
    return synced
