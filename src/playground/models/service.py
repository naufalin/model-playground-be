from __future__ import annotations

from typing import Any

import httpx
from fastapi import HTTPException, status

from playground.db.models import LlmModel
from playground.db.repos.model_repo import ModelRepo
from playground.models.schemas import ModelCreate
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


def _runtime_model_values(provider_key: str, model: dict[str, Any]) -> dict[str, Any] | None:
    provider = str(model.get("provider") or provider_key).lower()
    model_name = model.get("model_id")
    if not isinstance(model_name, str) or not model_name:
        return None
    return {
        "runtime_model_id": model.get("id") if isinstance(model.get("id"), int) else None,
        "provider": provider,
        "model_name": model_name,
        "display_name": str(model.get("name") or model_name),
        "is_active": bool(model.get("enabled", True)),
        "supports_reasoning": bool(model.get("supports_reasoning", False)),
        "sort_order": int(model.get("sort_order") or 0),
        "config_json": model.get("config") if isinstance(model.get("config"), dict) else None,
    }


def _runtime_error_detail(exc: httpx.HTTPStatusError) -> str:
    try:
        detail = exc.response.json().get("detail")
    except ValueError:
        detail = None
    if isinstance(detail, str):
        return detail
    return exc.response.text or "Agent runtime request failed"


def _raise_runtime_error(exc: httpx.HTTPStatusError) -> None:
    code = exc.response.status_code
    if code in (400, 409):
        raise HTTPException(status_code=code, detail=_runtime_error_detail(exc)) from exc
    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail=_runtime_error_detail(exc),
    ) from exc


async def sync_runtime_models(repo: ModelRepo, runtime: AgentRuntimeClient) -> tuple[int, int]:
    """Sync model metadata from agent runtime into the local registry."""
    payload = await runtime.list_models()
    synced = 0
    seen_models: set[tuple[str, str]] = set()
    for provider_key, model in _iter_runtime_models(payload):
        values = _runtime_model_values(provider_key, model)
        if values is None:
            continue
        await repo.upsert_runtime_model(**values)
        seen_models.add((values["provider"], values["model_name"]))
        synced += 1
    deactivated = await repo.deactivate_missing_runtime_models(seen_models)
    return synced, deactivated


async def create_runtime_model(
    body: ModelCreate,
    repo: ModelRepo,
    runtime: AgentRuntimeClient,
) -> LlmModel:
    """Create a runtime model and mirror it into the local registry."""
    try:
        model = await runtime.create_model(
            provider=body.provider,
            model_id=body.model_id,
            name=body.name,
            enabled=body.enabled,
            supports_reasoning=body.supports_reasoning,
            sort_order=body.sort_order,
            config=body.config,
        )
    except httpx.HTTPStatusError as exc:
        _raise_runtime_error(exc)
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not reach agent runtime",
        ) from exc
    values = _runtime_model_values(body.provider, model)
    if values is None:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Agent runtime returned an invalid model payload",
        )
    return await repo.upsert_runtime_model(**values)
