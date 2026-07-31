from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from playground.models.service import _fetch_openrouter_catalog
from playground.pricing.schemas import ModelPriceOut, PricingCatalogOut

OPENAI_PRICING_URL = "https://developers.openai.com/api/docs/pricing"
OPENROUTER_MODELS_URL = "https://openrouter.ai/docs/guides/overview/models"

# Standard, short-context API pricing in USD per 1M tokens.
# Keep the effective timestamp visible to clients whenever this snapshot changes.
OPENAI_PRICE_SNAPSHOT_AT = datetime(2026, 7, 31, tzinfo=UTC)
OPENAI_STANDARD_RATES: dict[str, tuple[float, float | None, float]] = {
    "gpt-5.6-sol": (5.0, 0.5, 30.0),
    "gpt-5.6-terra": (2.0, 0.2, 12.0),
    "gpt-5.6-luna": (0.2, 0.02, 1.2),
    "gpt-5.5": (5.0, 0.5, 30.0),
    "gpt-5.4": (2.5, 0.25, 15.0),
    "gpt-5.4-mini": (0.75, 0.075, 4.5),
    "gpt-5.4-nano": (0.2, 0.02, 1.25),
    "gpt-5.2": (1.75, 0.175, 14.0),
    "gpt-5.1": (1.25, 0.125, 10.0),
    "gpt-5": (1.25, 0.125, 10.0),
    "gpt-5-mini": (0.25, 0.025, 2.0),
    "gpt-5-nano": (0.05, 0.005, 0.4),
    "gpt-4.1": (2.0, 0.5, 8.0),
    "gpt-4.1-mini": (0.4, 0.1, 1.6),
    "gpt-4.1-nano": (0.1, 0.025, 0.4),
    "gpt-4o": (2.5, 1.25, 10.0),
    "gpt-4o-mini": (0.15, 0.075, 0.6),
    "o3": (2.0, 0.5, 8.0),
    "o4-mini": (1.1, 0.275, 4.4),
}
OPENAI_LONG_CONTEXT_MODELS = {
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.5",
    "gpt-5.4",
}


def estimate_openai_cost(
    usage: dict[str, Any] | None,
    model_name: str,
) -> dict[str, Any] | None:
    if (
        not usage
        or usage.get("cancelled") is True
        or isinstance(usage.get("cost"), dict)
        or not _is_non_negative_number(usage.get("input_tokens"))
        or not _is_non_negative_number(usage.get("output_tokens"))
    ):
        return usage
    rates = OPENAI_STANDARD_RATES.get(model_name)
    if rates is None:
        return usage
    input_tokens = _non_negative_number(usage.get("input_tokens"))
    if model_name in OPENAI_LONG_CONTEXT_MODELS and input_tokens >= 272_000:
        return usage
    output_tokens = _non_negative_number(usage.get("output_tokens"))
    cached_tokens = min(
        _non_negative_number(usage.get("cached_tokens")),
        input_tokens,
    )
    input_rate, cached_rate, output_rate = rates
    uncached_tokens = input_tokens - cached_tokens
    input_usd = uncached_tokens * input_rate / 1_000_000
    cached_input_usd = (
        cached_tokens * cached_rate / 1_000_000 if cached_rate is not None else 0.0
    )
    output_usd = output_tokens * output_rate / 1_000_000
    result = dict(usage)
    result["cost"] = {
        "amount_usd": input_usd + cached_input_usd + output_usd,
        "status": "estimated",
        "source": "catalog",
        "input_usd": input_usd,
        "cached_input_usd": cached_input_usd,
        "output_usd": output_usd,
        "pricing_snapshot_at": OPENAI_PRICE_SNAPSHOT_AT.isoformat(),
        "pricing_tier": "standard-short-context-assumption",
    }
    return result


async def pricing_catalog() -> PricingCatalogOut:
    refreshed_at = datetime.now(UTC)
    models = [
        ModelPriceOut(
            provider="openai",
            model_name=model_name,
            display_name=model_name,
            input_per_million_usd=rates[0],
            cached_input_per_million_usd=rates[1],
            output_per_million_usd=rates[2],
            rate_kind="exact",
            source_url=OPENAI_PRICING_URL,
            refreshed_at=OPENAI_PRICE_SNAPSHOT_AT,
        )
        for model_name, rates in OPENAI_STANDARD_RATES.items()
    ]
    try:
        entries = await _fetch_openrouter_catalog()
    except Exception:  # noqa: BLE001
        entries = []
    models.extend(
        parsed
        for entry in entries
        if (parsed := _openrouter_price(entry, refreshed_at)) is not None
    )
    return PricingCatalogOut(models=models, refreshed_at=refreshed_at)


def _openrouter_price(
    entry: dict[str, Any],
    refreshed_at: datetime,
) -> ModelPriceOut | None:
    model_name = entry.get("id")
    pricing = entry.get("pricing")
    if not isinstance(model_name, str) or not isinstance(pricing, dict):
        return None
    input_rate = _per_million(pricing.get("prompt"))
    output_rate = _per_million(pricing.get("completion"))
    cached_rate = _per_million(pricing.get("input_cache_read"))
    has_rate = input_rate is not None or output_rate is not None
    return ModelPriceOut(
        provider="openrouter",
        model_name=model_name,
        display_name=str(entry.get("name") or model_name),
        input_per_million_usd=input_rate,
        cached_input_per_million_usd=cached_rate,
        output_per_million_usd=output_rate,
        rate_kind="from" if has_rate else "unavailable",
        source_url=OPENROUTER_MODELS_URL,
        refreshed_at=refreshed_at,
    )


def _per_million(value: Any) -> float | None:
    try:
        rate = float(value)
    except (TypeError, ValueError):
        return None
    return rate * 1_000_000 if rate >= 0 else None


def _non_negative_number(value: Any) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool) and value >= 0:
        return float(value)
    return 0.0


def _is_non_negative_number(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and value >= 0
    )
