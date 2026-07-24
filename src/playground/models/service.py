from __future__ import annotations

import time
from typing import Any

import httpx
from fastapi import HTTPException, status

from playground.db.connection import Database
from playground.db.models import LlmModel
from playground.db.repos.model_repo import ModelRepo
from playground.models.schemas import ModelCreate, OpenRouterAvailableModel
from playground.runtime.client import AgentRuntimeClient

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
_OPENROUTER_CACHE_TTL_SECONDS = 300.0
_openrouter_cache: tuple[float, list[OpenRouterAvailableModel]] | None = None


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


async def sync_runtime_models(db: Database, runtime: AgentRuntimeClient) -> tuple[int, int]:
    """Sync model metadata from agent runtime into the local registry."""
    payload = await runtime.list_models()
    synced = 0
    seen_models: set[tuple[str, str]] = set()
    async with db.session() as session:
        repo = ModelRepo(session)
        for provider_key, model in _iter_runtime_models(payload):
            values = _runtime_model_values(provider_key, model)
            if values is None:
                continue
            await repo.upsert_runtime_model(**values)
            seen_models.add((values["provider"], values["model_name"]))
            synced += 1
        deactivated = await repo.deactivate_missing_runtime_models(seen_models)
    return synced, deactivated


def _parse_openrouter_model(entry: dict[str, Any]) -> OpenRouterAvailableModel | None:
    model_id = entry.get("id")
    if not isinstance(model_id, str) or not model_id:
        return None
    supported = entry.get("supported_parameters")
    supported_params = supported if isinstance(supported, list) else []
    context_length = entry.get("context_length")
    return OpenRouterAvailableModel(
        id=model_id,
        name=str(entry.get("name") or model_id),
        context_length=context_length if isinstance(context_length, int) else None,
        supports_reasoning=any(
            param in supported_params for param in ("reasoning", "include_reasoning")
        ),
    )


async def _fetch_openrouter_catalog() -> list[dict[str, Any]]:
    async with httpx.AsyncClient(timeout=httpx.Timeout(15)) as client:
        resp = await client.get(OPENROUTER_MODELS_URL)
        resp.raise_for_status()
        payload = resp.json()
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return []
    return [entry for entry in data if isinstance(entry, dict)]


async def list_openrouter_models() -> list[OpenRouterAvailableModel]:
    """Return the OpenRouter model catalog, trimmed for picker UIs and cached briefly."""
    global _openrouter_cache
    now = time.monotonic()
    if _openrouter_cache is not None:
        fetched_at, cached = _openrouter_cache
        if now - fetched_at < _OPENROUTER_CACHE_TTL_SECONDS:
            return cached
    try:
        entries = await _fetch_openrouter_catalog()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="OpenRouter catalog request failed",
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not reach OpenRouter",
        ) from exc
    models = [
        model
        for model in (_parse_openrouter_model(entry) for entry in entries)
        if model is not None
    ]
    _openrouter_cache = (now, models)
    return models


async def create_runtime_model(
    body: ModelCreate,
    db: Database,
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
    async with db.session() as session:
        return await ModelRepo(session).upsert_runtime_model(**values)
