from .valuation import calculate_valuation_score
from .growth import calculate_growth_score
from .momentum import calculate_momentum_score
from .quality import calculate_quality_score
from .risk import calculate_risk_score

SCORE_NAMES = ["valuation_score", "growth_score", "momentum_score", "quality_score", "risk_score"]


def calculate_all_scores(hist, info, financials) -> dict:
    return {
        "valuation_score": calculate_valuation_score(info),
        "growth_score": calculate_growth_score(financials),
        "momentum_score": calculate_momentum_score(hist),
        "quality_score": calculate_quality_score(info),
        "risk_score": calculate_risk_score(hist, info),
    }
