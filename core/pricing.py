"""Per-model pricing (USD per 1 M tokens — separate input / output rates).

Used by the agent loop to give real-time cost estimates after each turn.
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Price table  (USD per 1 M tokens)
# ─────────────────────────────────────────────────────────────────────────────

_PRICING: dict[str, dict[str, float]] = {
    # minimax
    "minimax-m2":         {"input": 0.40, "output": 1.60},
    "minimax-text":       {"input": 0.40, "output": 1.60},
    # kimi
    "kimi-k2":            {"input": 0.50, "output": 2.50},
    # qwen
    "qwen3.5":            {"input": 2.00, "output": 8.00},
    "qwq":                {"input": 2.00, "output": 8.00},
    # glm
    "glm5":               {"input": 0.50, "output": 2.00},
    "glm4":               {"input": 0.50, "output": 2.00},
    # llama nemotron
    "nemotron":           {"input": 1.00, "output": 4.00},
    "llama-3.1-nemotron": {"input": 1.00, "output": 4.00},
    # default fallback (blended estimate)
    "default":            {"input": 1.00, "output": 3.00},
}


def cost_usd(input_tokens: int, output_tokens: int, model: str) -> float:
    """Estimate cost in USD using separate input/output rates for the model."""
    model_lower = model.lower()
    rates = _PRICING["default"]
    for key, price in _PRICING.items():
        if key != "default" and key in model_lower:
            rates = price
            break
    return (
        input_tokens  * rates["input"]  / 1_000_000
        + output_tokens * rates["output"] / 1_000_000
    )
