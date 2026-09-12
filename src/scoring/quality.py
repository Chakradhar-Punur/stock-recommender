ROE_WEIGHT = 0.4
MARGIN_WEIGHT = 0.3
FCF_WEIGHT = 0.3


def calculate_roe_score(info: dict) -> float:
    roe = info.get("returnOnEquity")
    if roe is None:
        return 0.5
    return round(max(min(roe, 0.30), 0.0) / 0.30, 4)


def calculate_margin_score(info: dict) -> float:
    margin = info.get("profitMargins")
    if margin is None:
        return 0.5
    return round(max(min(margin, 0.25), 0.0) / 0.25, 4)


def calculate_fcf_score(info: dict) -> float:
    fcf, revenue = info.get("freeCashflow"), info.get("totalRevenue")
    if fcf is None or not revenue:
        return 0.5
    fcf_margin = fcf / revenue
    return round(max(min(fcf_margin, 0.20), 0.0) / 0.20, 4)


def calculate_quality_score(info: dict) -> float:
    combined = (
        calculate_roe_score(info) * ROE_WEIGHT
        + calculate_margin_score(info) * MARGIN_WEIGHT
        + calculate_fcf_score(info) * FCF_WEIGHT
    )
    return round(combined, 4)
