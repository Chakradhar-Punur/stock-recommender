import numpy as np


def _trend_score(values: list[float]) -> float:
    clean = [v for v in values if v is not None and not np.isnan(v)]
    if len(clean) < 4:
        return 0.5

    y = np.array(clean, dtype=float)
    scale = np.mean(np.abs(y))
    if scale == 0:
        return 0.5

    slope, _intercept = np.polyfit(np.arange(len(y)), y, 1)
    normalized_slope = slope / scale

    clipped = max(min(normalized_slope, 0.10), -0.10)
    return round(float((clipped + 0.10) / 0.20), 4)


def calculate_growth_score(financials: dict) -> float:
    revenue_score = _trend_score(financials.get("quarterly_revenue", []))
    earnings_score = _trend_score(financials.get("quarterly_earnings", []))
    return round((revenue_score + earnings_score) / 2, 4)
