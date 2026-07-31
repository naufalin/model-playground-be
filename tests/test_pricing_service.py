from datetime import UTC, datetime

import pytest

from playground.pricing.service import _openrouter_price, estimate_openai_cost


def test_estimate_openai_cost_separates_cached_input():
    usage = {
        "input_tokens": 1_000_000,
        "cached_tokens": 200_000,
        "output_tokens": 100_000,
        "total_tokens": 1_100_000,
    }

    result = estimate_openai_cost(usage, "gpt-5.4-mini")

    assert result is not None
    assert result["cost"]["status"] == "estimated"
    assert result["cost"]["input_usd"] == pytest.approx(0.6)
    assert result["cost"]["cached_input_usd"] == pytest.approx(0.015)
    assert result["cost"]["output_usd"] == pytest.approx(0.45)
    assert result["cost"]["amount_usd"] == pytest.approx(1.065)


def test_estimate_openai_cost_preserves_provider_actual_cost():
    usage = {
        "total_tokens": 10,
        "cost": {"amount_usd": 0.1, "status": "actual", "source": "provider"},
    }

    assert estimate_openai_cost(usage, "gpt-5.4-mini") is usage


def test_estimate_openai_cost_leaves_missing_or_cancelled_usage_unpriced():
    missing = {"total_tokens": 10}
    cancelled = {"cancelled": True, "input_tokens": 0, "output_tokens": 0}

    assert estimate_openai_cost(missing, "gpt-5.4-mini") is missing
    assert estimate_openai_cost(cancelled, "gpt-5.4-mini") is cancelled


def test_estimate_openai_cost_fails_closed_for_long_context_rate():
    usage = {"input_tokens": 272_000, "output_tokens": 1}

    assert estimate_openai_cost(usage, "gpt-5.4") is usage


def test_openrouter_catalog_converts_per_token_rates_to_per_million():
    result = _openrouter_price(
        {
            "id": "vendor/model",
            "name": "Vendor Model",
            "pricing": {
                "prompt": "0.000002",
                "completion": "0.000008",
                "input_cache_read": "0.0000002",
            },
        },
        datetime(2026, 7, 31, tzinfo=UTC),
    )

    assert result is not None
    assert result.input_per_million_usd == 2
    assert result.cached_input_per_million_usd == pytest.approx(0.2)
    assert result.output_per_million_usd == 8
    assert result.rate_kind == "from"
