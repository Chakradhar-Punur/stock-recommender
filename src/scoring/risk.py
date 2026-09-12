import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def calculate_volatility_score(hist: pd.DataFrame) -> float:
    if hist is None or hist.empty or "Close" not in hist.columns or len(hist) < 30:
        return 0.5

    daily_returns = hist["Close"].pct_change().dropna()
    if daily_returns.empty:
        return 0.5

    annualized_vol = daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    clipped = max(min(annualized_vol, 0.60), 0.15)
    return round(float(1 - (clipped - 0.15) / (0.60 - 0.15)), 4)


def calculate_beta_score(info: dict) -> float:
    beta = info.get("beta")
    if beta is None or beta < 0:
        return 0.5
    clipped = max(min(beta, 2.0), 0.5)
    return round(1 - (clipped - 0.5) / (2.0 - 0.5), 4)


def calculate_risk_score(hist: pd.DataFrame, info: dict) -> float:
    return round((calculate_volatility_score(hist) + calculate_beta_score(info)) / 2, 4)
