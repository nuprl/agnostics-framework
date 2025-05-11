#!/usr/bin/env python3
"""
Simple calculator for computing costs from LLM API token usage.

Usage:
  util_llm_cost.py <model_name> <input_tokens> <output_tokens>

Example:
  ./scripts/util_llm_cost.py openai/gpt-5.2 1000000 250000
"""

from __future__ import annotations

import argparse
from decimal import Decimal


# Prices are in USD per 1M tokens.
_PRICING_USD_PER_1M = {
    "openai/gpt-5.2": {"input": Decimal("1.75"), "output": Decimal("14")},
    "anthropic/claude-opus-4-6": {"input": Decimal("5"), "output": Decimal("25")},
}
_PRICING_USD_PER_1M["gpt-5.2"] = _PRICING_USD_PER_1M["openai/gpt-5.2"]
_PRICING_USD_PER_1M["claude-opus-4-6"] = _PRICING_USD_PER_1M["anthropic/claude-opus-4-6"]


def _fmt_money_usd(amount: Decimal) -> str:
    # Keep enough precision for small runs, but avoid noisy trailing zeros.
    q = amount.quantize(Decimal("0.000001"))
    s = format(q, "f").rstrip("0").rstrip(".")
    return f"${s if s else '0'}"


def main() -> int:
    p = argparse.ArgumentParser(
        description="Compute LLM API cost from input/output tokens.",
    )
    p.add_argument("model_name", help="Model name (e.g. openai/gpt-5.2).")
    p.add_argument("input_tokens", type=int, help="Number of input tokens.")
    p.add_argument("output_tokens", type=int, help="Number of output tokens.")
    args = p.parse_args()

    if args.input_tokens < 0 or args.output_tokens < 0:
        raise SystemExit("Error: token counts must be non-negative integers.")

    pricing = _PRICING_USD_PER_1M.get(args.model_name)
    if pricing is None:
        known = ", ".join(sorted(_PRICING_USD_PER_1M))
        raise SystemExit(f"Error: unknown model {args.model_name!r}. Known: {known}")

    one_million = Decimal("1000000")
    cost = (
        (Decimal(args.input_tokens) / one_million) * pricing["input"]
        + (Decimal(args.output_tokens) / one_million) * pricing["output"]
    )

    print(_fmt_money_usd(cost))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

