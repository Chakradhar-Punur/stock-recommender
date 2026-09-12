SCORE_LABELS = {
    "valuation_score": "Valuation",
    "growth_score": "Growth",
    "momentum_score": "Momentum",
    "quality_score": "Quality",
    "risk_score": "Risk",
}

SCORE_DESCRIPTIONS = {
    "valuation_score": (
        "cheap relative to earnings, growth, and debt load",
        "expensive relative to earnings, growth, and debt load",
    ),
    "growth_score": (
        "revenue and earnings have been trending up",
        "revenue and earnings have been flat or declining",
    ),
    "momentum_score": (
        "strong recent price trend and buying pressure",
        "weak recent price trend and buying pressure",
    ),
    "quality_score": (
        "strong returns on equity, margins, and free cash flow",
        "weak returns on equity, margins, or free cash flow",
    ),
    "risk_score": (
        "low volatility and market-correlated risk",
        "high volatility and market-correlated risk",
    ),
}


def explain(scores: dict) -> dict:
    best_name = max(scores, key=lambda k: scores[k])
    worst_name = min(scores, key=lambda k: scores[k])

    best_desc, _ = SCORE_DESCRIPTIONS[best_name]
    _, worst_desc = SCORE_DESCRIPTIONS[worst_name]

    return {
        "key_driver": f"{SCORE_LABELS[best_name]} ({scores[best_name]:.2f}) — {best_desc}.",
        "biggest_risk": f"{SCORE_LABELS[worst_name]} ({scores[worst_name]:.2f}) — {worst_desc}.",
    }
