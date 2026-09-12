# Stock Recommender

An end-to-end ML system that scores a stock ticker across 5 fundamental/
technical dimensions, combines them with a trained XGBoost model into a
**BUY / SELL** call with a confidence percentage, and serves it through a
FastAPI backend and a React frontend. Built as a portfolio project to show
a complete pipeline — data → features → training → backtesting → serving →
UI — not just a notebook, with every design tradeoff documented rather than
hidden.

Not investment advice. Backtest results below include the honest
limitations, not just the flattering numbers.

## Architecture

```
data_collection.py  → fetch price + fundamentals (yfinance)
        ↓
scoring/             → 5 scores: valuation, growth, momentum, quality, risk
        ↓
build_training_data.py → sample tickers, label by forward return
        ↓
train_model.py       → train XGBoost (time-based split)
        ↓
backtest.py           → walk-forward backtest vs S&P 500
        ↓
inference.py          → score any ticker live
        ↓
explain.py            → key driver / biggest risk text
        ↓
app.py                → FastAPI: /score, /health
        ↓
frontend/             → React UI
```

`baseline_model.py` is a separate, deliberately "dumb" rule-based
BUY/HOLD/SELL (simple average of momentum + valuation, fixed thresholds) —
not the ML model. It exists purely as a comparison bar: "beats a naive
average by N points" is a defensible claim on its own.

## Project structure

```
stock-recommender/
├── src/
│   ├── data_collection.py     # fetch price history + fundamentals
│   ├── features.py            # momentum / valuation primitives (shared)
│   ├── scoring/                # the 5 component scores
│   │   ├── valuation.py
│   │   ├── growth.py
│   │   ├── momentum.py
│   │   ├── quality.py
│   │   └── risk.py
│   ├── baseline_model.py      # rule-based BUY/HOLD/SELL (comparison only)
│   ├── build_training_data.py
│   ├── train_model.py
│   ├── backtest.py            # walk-forward backtest
│   ├── inference.py
│   ├── explain.py             # key-driver / biggest-risk text
│   └── app.py                 # FastAPI server
├── frontend/                  # React (Vite) — see frontend/README.md
├── docs/
│   └── CODE_NOTES.md
├── data/processed/            # generated training_data.csv
├── models/                    # generated xgboost_v1.pkl
└── requirements.txt
```

## Setup

**No API keys or environment variables are required** — `yfinance` pulls
public Yahoo Finance data with no auth, so there's no `.env` file in this
project.

**Backend** — virtual env already at `./venv` (Python 3.11):

```bash
python3.11 -m venv venv && venv/bin/pip install -r requirements.txt
```

> **Use `python3.11` explicitly, not bare `python3`.** macOS ships an old
> system Python (3.9) whose pip can't resolve `pandas==3.0.5` (no wheel
> published for 3.9) — you'll get a confusing "No matching distribution"
> error if you create the venv with the wrong interpreter. Verified: a
> fresh venv built with `python3.11` (Homebrew: `brew install python@3.11`)
> installs every pinned version in `requirements.txt` with no errors.

> macOS only: if XGBoost fails to import with a `libomp.dylib` error, run
> `brew install libomp` — it needs Apple's OpenMP runtime, which isn't
> bundled.

**Frontend**:

```bash
cd frontend && npm install
```

## Running it

Build the training data, train, and backtest (only needed once, or to
refresh with newer data):

```bash
venv/bin/python src/build_training_data.py
venv/bin/python src/train_model.py
venv/bin/python src/backtest.py
```

> `data/processed/` and `models/` are gitignored on purpose — they're
> per-machine artifacts, not shared via git. If you're setting this up on
> a second machine, run the commands above there too rather than copying
> the `.pkl` file over, so the model is trained and loaded by the same
> XGBoost version installed on that machine.

Then, in two terminals:

```bash
cd src && ../venv/bin/uvicorn app:app --reload --port 8000
```

```bash
cd frontend && npm run dev
```

Open the URL Vite prints (usually `http://localhost:5173`), type a ticker
(e.g. `MSFT`), and click Analyze. Or hit the API directly:

```bash
curl http://localhost:8000/score/AAPL
curl http://localhost:8000/health
```

`http://localhost:8000/docs` gives an interactive Swagger UI for the API.

## Backtest results (walk-forward, 2020-09 to 2025-09)

Run via `src/backtest.py`, which retrains the model from scratch at every
historical snapshot date (not one fixed 80/20 split) — 210 out-of-sample
predictions across ~21 independent folds.

| Metric | Value |
|---|---|
| Overall accuracy | 0.605 |
| BUY signals issued | 157 / 210 |
| Win rate on BUY signals | 0.796 |
| Annual Sharpe ratio (honest, non-overlapping) | 0.992 |
| Annual max drawdown (honest) | -0.009 |
| Compounded total return (honest, 5 years) | +412.8% |
| S&P 500 buy-and-hold, same period | +127.2% |

**Year-by-year** (where the model struggled, on purpose surfaced rather
than averaged away):

| Year | Accuracy | Win rate | S&P 500 return |
|---|---|---|---|
| 2020 | 0.700 | 1.000 | +12.4% |
| 2021 | **0.300** | 0.394 | +28.8% |
| 2022 | 0.575 | 0.769 | -20.0% |
| 2023 | 0.800 | 0.968 | +24.7% |
| 2024 | 0.675 | 0.935 | +24.0% |
| 2025 | 0.633 | 0.857 | +16.6% |

2021 is the model's worst year — a real, explainable finding: 2021
snapshots predict 12 months forward into the 2022 rate-hike bear market, a
regime shift the model's training data hadn't seen examples of yet at that
point. 2022's own snapshots do noticeably better (0.575) because by then
some bear-market examples existed in training.

## Limitations

- **This is a small dataset.** ~290 rows (10 tickers × ~29 quarterly
  snapshots) trains a model with only ~230 rows in any given split. Phase 2
  (5 features vs. the original 2) improved a single train/test split's
  accuracy from 0.533 → 0.650, but that exact number is sensitive to which
  week you happen to run it — a genuine limitation the walk-forward backtest
  above exists specifically to average out.
- **Two Sharpe ratios are reported on purpose.** A naive calculation that
  compounds the strategy's quarterly returns sequentially double- and
  triple-counts overlapping 12-month-forward windows and reports an
  inflated Sharpe of 2.17. The honest calculation uses one non-overlapping
  return per calendar year (Sharpe 0.99) — correct methodology, but only 6
  data points, so treat it as directionally useful, not precise.
- **Valuation/growth/quality scores aren't point-in-time.** Free
  `yfinance` data only exposes *current* fundamentals, not historical P/E,
  quarterly financials, or ROE as they stood on a past date — so every
  historical training row for a ticker reuses that ticker's *current*
  values for these three scores. Only momentum and risk's volatility
  component are genuinely computed from the correct historical price
  window. A production version would pull point-in-time fundamentals from
  a paid vendor (Compustat, FactSet, etc.).
- **Binary model, not the baseline's 3-way logic.** `baseline_model.py`
  outputs BUY/HOLD/SELL from fixed rules; the trained model's label is
  binary (beat +10%/12mo or not), so `inference.py` only ever returns
  BUY/SELL.
- **In-memory cache, not distributed.** `app.py`'s cache is a plain dict —
  resets on restart, doesn't share state across multiple server processes.
  Fine for a local demo, a real explicit tradeoff not to build Redis for.
- **No deployment yet.** Runs locally only — no Docker, no cloud hosting.

## Tech stack

**Backend:** Python, yfinance, pandas, numpy, scikit-learn, XGBoost, FastAPI.
**Frontend:** React (Vite), recharts, plain CSS (light/dark mode via CSS
custom properties).
