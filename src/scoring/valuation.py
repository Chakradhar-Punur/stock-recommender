from features import calculate_valuation_score as _pe_score

PE_WEIGHT = 0.4
PEG_WEIGHT = 0.4
DEBT_WEIGHT = 0.2


def calculate_peg_score(info: dict) -> float:
    peg = info.get("pegRatio")
    pe = info.get("trailingPE")
    earnings_growth = info.get("earningsGrowth")

    if peg is None and pe is not None and earnings_growth:
        growth_pct = earnings_growth * 100
        if growth_pct > 0:
            peg = pe / growth_pct

    if peg is None or peg <= 0:
        return 0.5

    clipped = max(min(peg, 3.0), 0.5)
    return round(1 - (clipped - 0.5) / (3.0 - 0.5), 4)


def calculate_debt_score(info: dict) -> float:
    de = info.get("debtToEquity")
    if de is None or de < 0:
        return 0.5

    de_ratio = de / 100
    clipped = max(min(de_ratio, 2.0), 0.0)
    return round(1 - clipped / 2.0, 4)


def calculate_valuation_score(info: dict) -> float:
    combined = (
        _pe_score(info) * PE_WEIGHT
        + calculate_peg_score(info) * PEG_WEIGHT
        + calculate_debt_score(info) * DEBT_WEIGHT
    )
    return round(combined, 4)
