# Stock Recommender

A small end-to-end machine learning system that scores a stock ticker and
returns a **BUY / SELL** call with a confidence percentage — plus a web UI
to demo it. Built as a portfolio project to show a complete ML pipeline
(raw data → features → training → serving → UI), not just a notebook.

> **Not investment advice.** This is a demonstration of an ML engineering
> pipeline. See [Honest limitations](#honest-limitations) below.

## How it works

```
data_collection.py   → pulls price history + fundamentals (yfinance)
        ↓
features.py           → turns that into two 0–1 scores (momentum, valuation)
        ↓
build_training_data.py → samples 10 large-cap tickers every ~3 months over
                          ~7 years, labels each point: did the stock beat
                          +10% return over the following 12 months?
        ↓
train_model.py         → trains an XGBoost classifier with a TIME-BASED
                          train/test split (no lookahead bias)
        ↓
inference.py            → loads the trained model, scores any ticker live
        ↓
app.py                  → FastAPI wrapper exposing GET /score/{ticker}
        ↓
frontend/                → React (Vite) UI that calls the API and shows a
                          color-coded result card
```

Each stage does one job and hands its output to the next — see
[src/](src/) for the individual files, each with docstrings explaining
*why*, not just what.

## Project structure

```
stock-recommender/
├── src/                      # Python backend pipeline
│   ├── data_collection.py    # fetch price history + fundamentals
│   ├── features.py           # momentum / valuation scoring
│   ├── baseline_model.py     # rule-based BUY/HOLD/SELL (comparison baseline)
│   ├── build_training_data.py
│   ├── train_model.py
│   ├── inference.py
│   └── app.py                # FastAPI server
├── frontend/                 # React (Vite) web UI
├── data/processed/           # generated training_data.csv (gitignored)
├── models/                   # generated xgboost_v1.pkl (gitignored)
└── requirements.txt
```

## Setup

**Backend** (Python 3.13, virtual env already at `./venv`):

```bash
python3 -m venv venv && venv/bin/pip install -r requirements.txt
```

> macOS only: if XGBoost fails to import with a `libomp.dylib` error, run
> `brew install libomp` — it needs Apple's OpenMP runtime, which isn't
> bundled.

**Frontend**:

```bash
cd frontend && npm install
```

## Running it

Rebuild the training data and model (only needed once, or to refresh with
newer data):

```bash
venv/bin/python src/build_training_data.py
venv/bin/python src/train_model.py
```

Then, in two terminals:

```bash
cd src && ../venv/bin/uvicorn app:app --reload --port 8000
```

```bash
cd frontend && npm run dev
```

Open the URL Vite prints (usually `http://localhost:5173`), type a ticker
(e.g. `MSFT`), and click Analyze.

## Honest limitations

- **~50% test accuracy.** With only two features and ~290 training
  examples, the model is close to a coin flip on held-out data. The
  point of this stage was proving the *pipeline* is correct (proper
  time-based split, no leakage) — not a profitable strategy.
- **Valuation score isn't point-in-time.** Free `yfinance` data only
  exposes *today's* P/E ratio, so every historical training row for a
  ticker uses that ticker's current P/E, not what it actually was back
  then. Flagged in [build_training_data.py](src/build_training_data.py).
- **Binary model, not the baseline's 3-way logic.** `baseline_model.py`
  outputs BUY/HOLD/SELL from fixed rules; the trained model's label is
  binary (beat +10%/12mo or not), so `inference.py` only ever returns
  BUY/SELL.

## Tech stack

Python, yfinance, pandas, scikit-learn, XGBoost, FastAPI, React (Vite).
