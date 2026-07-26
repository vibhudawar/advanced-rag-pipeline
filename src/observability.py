"""Lightweight observability helpers (WIN 8).

Two layers:
  1. LangSmith tracing — enabled purely by env (`LANGCHAIN_TRACING_V2=true` + `LANGSMITH_API_KEY`);
     the production pipeline entrypoints are decorated with `@traceable`, so each request shows as
     one nested trace (retrieve → rerank → gate → generate) with latency/tokens/cost in the
     LangSmith UI. No code change needed to toggle it — absent the env vars, `@traceable` is a
     near-no-op.
  2. In-app per-answer metrics — token usage is aggregated via LangChain's usage callback; this
     module turns that into a compact dict (tokens + USD cost) persisted to `messages.metadata`
     and surfaced under each answer in the UI.

Prices are USD per 1M tokens (approximate list prices; override via env if they drift).
"""

from __future__ import annotations

import os

# USD per 1,000,000 tokens. Only models this app actually calls need entries.
_PRICES: dict[str, tuple[float, float]] = {  # model -> (input, output)
    "gpt-5.4-nano": (0.20, 1.25),   # default generator (lite, cost-efficient)
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),        # eval judge (judge != generator)
}


def _price(model: str) -> tuple[float, float]:
    for name, price in _PRICES.items():
        if model.startswith(name):
            return price
    return (0.0, 0.0)  # unknown model → don't guess a cost


def summarize_usage(usage_metadata: dict) -> dict:
    """Collapse LangChain's per-model usage dict into flat metrics.

    `usage_metadata` looks like {"gpt-4o-mini": {"input_tokens": N, "output_tokens": M, ...}}.
    Returns input/output/total tokens, USD cost, and the model(s) involved.
    """
    input_tokens = output_tokens = 0
    cost = 0.0
    models: list[str] = []
    for model, usage in (usage_metadata or {}).items():
        inp = int(usage.get("input_tokens", 0) or 0)
        out = int(usage.get("output_tokens", 0) or 0)
        input_tokens += inp
        output_tokens += out
        p_in, p_out = _price(model)
        cost += inp / 1_000_000 * p_in + out / 1_000_000 * p_out
        models.append(model)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost_usd": round(cost, 6),
        "model": ", ".join(models) if models else None,
    }


def langsmith_enabled() -> bool:
    """Whether LangSmith tracing is switched on via env (for logging/health only)."""
    return os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
