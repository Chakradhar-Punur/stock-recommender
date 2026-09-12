import pandas as pd

from features import calculate_momentum_score as _price_trend_score

RSI_PERIOD = 14


def calculate_rsi(hist: pd.DataFrame, period: int = RSI_PERIOD) -> float | None:
    if hist is None or hist.empty or "Close" not in hist.columns or len(hist) < period + 1:
        return None

    delta = hist["Close"].diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    last_gain, last_loss = avg_gain.iloc[-1], avg_loss.iloc[-1]
    if last_loss == 0:
        return 100.0

    rs = last_gain / last_loss
    return round(float(100 - (100 / (1 + rs))), 2)


def calculate_rsi_score(hist: pd.DataFrame) -> float:
    rsi = calculate_rsi(hist)
    return 0.5 if rsi is None else round(rsi / 100, 4)


def calculate_momentum_score(hist: pd.DataFrame) -> float:
    return round((_price_trend_score(hist) + calculate_rsi_score(hist)) / 2, 4)
