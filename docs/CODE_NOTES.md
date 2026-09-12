# Code Notes

Design rationale for every file in `src/`. The source files themselves are
kept comment-free by preference — this doc is where the "why" behind each
decision lives instead, so it's easy to review as a whole and easy to talk
through in an interview.

## `data_collection.py`

Fetches raw price history and fundamentals from `yfinance` for one ticker.

- **Broad `except Exception`** around the whole fetch: `yfinance` can fail in
  many ways (network error, rate limiting, malformed response), and this is a
  leaf function nothing downstream should ever have to guard against — it
  returns an empty-but-valid shape (`{"history": DataFrame(), "info": {},
  "financials": {}}`) instead of raising, so callers never need a special
  "did fetching even work" branch.
- **`.info` is fetched defensively in its own try/except**: it's the flakiest
  part of `yfinance` (a scraped dict whose fields vary and can go missing
  entirely across tickers/versions). A failure here degrades to
  price-history-only rather than failing the whole ticker.
- **Only specific fields are pulled out of the raw `.info` dict**, not the
  whole thing — this is the single place all five scoring modules' data
  dependencies are declared, so it's obvious at a glance what the model
  actually depends on.
- **`_get_quarterly_financials`** pulls `stock.quarterly_financials` for the
  growth score's trend regression. Line-item labels ("Net Income" vs "Net
  Income Common Stockholders") aren't fully consistent across tickers, so
  `_extract_row` tries a list of candidate labels in order and only gives up
  once all of them miss — an earlier version of this bug returned early on
  the *first* miss, silently defeating the whole point of having fallback
  labels.
- **Columns come back newest-first from `yfinance`**; `_extract_row` reverses
  them to oldest-first, since a regression needs a consistent time direction
  to get the growth slope's sign right.

## `features.py`

Two original scoring primitives, kept around because `baseline_model.py`
and several `scoring/` modules reuse them rather than re-deriving the same
math twice.

- **`calculate_momentum_score`**: 30-trading-day price change, clipped to
  ±20% and mapped linearly onto [0, 1]. 30 days is short enough to reflect a
  real recent trend, long enough to filter single-day noise. The ±20% clip
  exists because occasional outliers (earnings-day spikes) would otherwise
  swamp the score; days rarely move a stock more than that in a month.
- **`calculate_valuation_score`**: trailing P/E mapped onto [0, 1] over a
  [10, 40] band (10 = cheap → score 1.0, 40 = expensive → score 0.0). A
  negative or missing P/E (e.g. negative earnings) scores neutral (0.5)
  rather than being penalized — a negative P/E doesn't mean "infinitely
  expensive," it means the ratio is undefined.
- Both functions return **0.5 (neutral) rather than raising** when there
  isn't enough data — callers shouldn't have to special-case short
  histories or missing fields.

## `baseline_model.py`

A deliberately dumb, rule-based BUY/HOLD/SELL — **not** the ML model. It
exists purely as a comparison point: "our model beats a simple average-based
rule by N points" is a defensible claim; "our model gets 62% accuracy" on
its own is not.

- Combines momentum + valuation with a plain **average** and **untuned,
  round-number thresholds** (0.65 / 0.35) on purpose. If the baseline were
  already clever, it would set an unfairly high bar and muddy the
  comparison — the point is to be as transparent and "dumb" as possible.

## `scoring/` — the five component scores

Each module follows the same shape as `features.py`: pure functions (data
in, float in [0, 1] out), each with an explicit "what if the data is
missing" fallback (usually 0.5, neutral) instead of raising. `scoring/
__init__.py` exposes `SCORE_NAMES` — the single ordered list of the five
feature names that both `build_training_data.py` and `inference.py` import,
rather than each hardcoding its own copy. This is the fix for train/serve
skew: a model trained on `[A, B, C]` fed inputs as `[B, A, C]` at serving
time is a real, common, silent bug class in ML systems, and a shared
constant is how you make it structurally impossible.

### `valuation.py`
P/E (reused from `features.py`) + PEG + debt-to-equity, weighted 0.4/0.4/0.2.
- **PEG** (P/E relative to earnings growth) exists because a high P/E can
  still be "cheap" if earnings are growing fast enough to justify it — P/E
  alone can't distinguish an expensive stagnant company from an expensive
  fast grower. Prefers `yfinance`'s own `pegRatio` field, falls back to
  computing `P/E ÷ (earningsGrowth × 100)` when that field is missing (it
  isn't populated for every ticker).
- **Debt-to-equity** is grouped under valuation, not risk, because heavier
  leverage makes a dollar of equity worth less/riskier per dollar of
  earnings — a valuation concern, not purely a volatility one. `yfinance`
  reports it as a percentage (150.0 = D/E of 1.5), so it's divided by 100
  before scoring.
- Weights (0.4/0.4/0.2) are a reasoned starting point, not backtested or
  tuned — same "flag it, don't hide it" spirit as the baseline's thresholds.

### `growth.py`
Revenue-trend and earnings-trend scores from trailing quarterly financials,
combined with equal weight.
- Uses a **simple linear regression slope**, not a sequence model (LSTM,
  etc.). `yfinance` only exposes ~5 quarters of history per ticker — nowhere
  near enough to fit a sequence model without instantly overfitting. A slope
  a reviewer can sanity-check by eye is a better tradeoff here than a complex
  model with no way to validate it on this little data.
- The slope is **normalized by the series' mean magnitude** before scoring
  (`slope / mean(|values|)`), turning it into a *growth rate* (e.g. 4% per
  quarter) rather than an absolute dollar change — otherwise a $3T company
  and a $50B one could never be compared on the same scale.
- Returns 0.5 (neutral) with fewer than 4 data points or an all-zero series,
  rather than fitting a meaningless line to almost nothing.

### `momentum.py`
Blends the existing 30-day price-trend score with a 14-day RSI (Relative
Strength Index).
- **RSI uses Wilder's original smoothing** (an exponential moving average of
  gains/losses via `.ewm(alpha=1/period)`), the standard definition most
  charting platforms use — chosen so this RSI value matches what you'd see
  for the same ticker/period elsewhere, not some ad hoc simple-average
  variant.
- **High RSI is treated as a *positive* momentum signal here**, not the
  "overbought, about to reverse" warning a mean-reversion strategy would
  read it as. That's a deliberate framing choice consistent with
  `calculate_momentum_score`'s "recent strength is good" philosophy — worth
  noting explicitly since it's the opposite of RSI's most common textbook use.
- Blending a trend measure with an oscillator catches things pure trend
  alone can miss — e.g. a stock that's up a lot over 30 days but has just
  started rolling over in the last few sessions.

### `quality.py`
ROE + net profit margin + free-cash-flow margin, weighted 0.4/0.3/0.3.
- **FCF margin is derived, not a direct field**: `freeCashflow ÷
  totalRevenue`. FCF exists as a separate signal from net income
  specifically because a company can report positive net income while
  burning cash (aggressive capex, working-capital swings) — it's a
  harder-to-fake quality signal than net income alone.
- Each sub-score clips to a band chosen from what's typical/exceptional for
  a large-cap (ROE: 0–30%, margin: 0–25%, FCF margin: 0–20%) — above the top
  of each band is "elite" (AAPL/MSFT-tier), and clipping there avoids one
  outlier metric dominating the composite.

### `risk.py`
Volatility + beta, equal weight.
- **Higher score = lower risk**, matching the "higher is better for BUY"
  direction the other four scores use. This isn't required by XGBoost (it
  can learn any relationship, in any direction, per feature) but keeps each
  of the five scores individually interpretable when explaining a
  recommendation later — "risk_score was low" should always mean "this was
  a point against BUY," consistently across every score.
- **Volatility** is the annualized std dev of daily returns
  (`daily_std × √252`), which puts it on the familiar "X% per year" scale
  quoted in finance rather than a hard-to-interpret daily-return std dev.
- **Beta** uses Yahoo's own precomputed value rather than being calculated
  from scratch — computing it ourselves would mean fetching and
  time-aligning S&P 500 history for every single ticker, real engineering
  effort that Yahoo's own long-window calculation already covers adequately
  for a project at this scale.

## `build_training_data.py`

Walks each ticker's price history, samples a snapshot every ~3 months over
8 years, and labels each snapshot: did the stock beat a +10% return over the
following 12 months?

- **No train/test split happens in this file.** It only assembles raw
  labeled data; the split (by date, no shuffling) happens later in
  `train_model.py`. Keeping that leakage-sensitive logic in exactly one
  place means it can't accidentally be implemented two different (and
  possibly inconsistent) ways.
- **Which scores are computed once per ticker vs. recomputed per
  snapshot** is a deliberate split based on what `yfinance`'s free tier
  actually allows:
  - `valuation_score`, `growth_score`, `quality_score` all depend on
    *current* fundamentals/quarterly financials — there's no free
    point-in-time history for these, so the same value is repeated across
    every historical row for a given ticker. This is a known, real
    limitation (not an oversight) — a production system would pull
    point-in-time fundamentals from a paid vendor (Compustat, FactSet, etc.).
  - `momentum_score` and risk's volatility component genuinely **can** be
    computed correctly at each historical snapshot, because full daily
    price history is already available and can be sliced up to any date
    (`hist_upto_T`) — so they are recomputed fresh at every sample point
    rather than taking the same static-fundamentals shortcut.
- **`_price_row_asof`** finds the most recent trading day at-or-before a
  given calendar date (markets are closed weekends/holidays, so you can't
  index by an arbitrary date directly). It uses `isinstance(row, pd.Series)`
  rather than `row is None` to narrow the type, since `.asof()`'s return
  type is ambiguous in pandas' stubs (it can return a `Series`, a
  `DataFrame`, or an `NaT`/`NA` sentinel depending on input shape).
- **The forward-return / label computation guards against a stale `asof`
  hit**: if there isn't actually `forward_months` of future data yet,
  `.asof()` would silently return the *most recent available* row instead of
  failing — which would fabricate a label from data that doesn't really
  represent 12 months forward. The check
  `hist.index.max() >= future_date - pd.Timedelta(days=10)` rejects that
  case explicitly.

## `train_model.py`

Trains an XGBoost classifier with a **time-based train/test split** — sorted
by date, no shuffling, one single cutoff date.

- **Why time-based, not random**: these are financial time-series samples.
  A stock's scores in Q1 2020 and its label (did it beat +10% by Q1 2021)
  are both shaped by that whole COVID-era macro window. A random split would
  put some Q1-2020-window rows in train and others from the *same* window in
  test — the model could then pick up "what generally happened in
  2020–2021" from train and score well on test for that reason alone, not
  because it learned a real predictive pattern. That's leakage in disguise.
  A live trading system can never train on the future, so the evaluation
  should respect that same constraint: train strictly on the past, test
  strictly on what comes after.
- **Split by cutoff *date*, not by row count**: splitting by row count could
  cut a single date's cross-section of ~10 tickers in half (some tickers
  land in train, others in test, for the *same* date) — that still leaks
  same-period information across the boundary. A single date cutoff keeps
  every date's full cross-section on one side only.
- **Shallow model (`max_depth=3`, `n_estimators=100`)** is intentional, not
  a default left untouched — with only ~230 training rows even after the
  5-feature expansion, a high-capacity model would overfit noise rather than
  learn signal.
- **Feature importance is printed** (XGBoost's "gain"-based importance —
  average split-quality improvement per feature) specifically to check the
  model is actually using all 5 scores rather than leaning on just one and
  ignoring the rest.
- **`PHASE_1_BASELINE_ACCURACY = 0.533`** is hardcoded from the original
  2-feature (momentum + valuation only) run, so this script can print a
  direct before/after comparison without needing to re-run the old model
  every time.

## `inference.py`

The serving layer: load the trained model once, then score any ticker on
demand.

- **`FEATURE_COLUMNS = SCORE_NAMES`**, imported from `scoring/__init__.py`
  rather than redeclared — this is what makes train/serve skew structurally
  impossible rather than just "something to remember to keep in sync."
- **Returns an `"error"` key instead of raising** on bad tickers or a
  missing model file, so a caller scoring a batch of tickers can skip
  failures without the whole run crashing — and so `app.py` has something
  concrete to translate into an HTTP error status.
- **The `scores` dict is included in the response** (not just the final
  recommendation) — needed for the API's planned feature-importance/
  explanation output in Phase 4, and useful on its own for debugging *why*
  a given ticker got the call it did.

## `backtest.py`

A **walk-forward** backtest: unlike `train_model.py`'s single 80/20 date
cutoff, this retrains the model from scratch at every historical snapshot
date, using only data strictly before it, then predicts on that date and
moves forward. This is what "backtest" should actually mean for a
time-series model — it produces ~20 independent out-of-sample train/test
folds instead of one, which is far more informative for both an honest
accuracy estimate and a real year-by-year breakdown (`train_model.py`'s
single split only ever tests on the *last* ~20% of dates, so it can't tell
you anything about, say, 2021 specifically).

- **`MIN_TRAIN_DATES = 8`** (~2 years) is a warm-up: the first 8 unique
  snapshot dates are used only for training, never scored, since predicting
  from a near-empty training set would just measure noise.
- **`build_model()` is imported from `train_model.py`** rather than
  reconstructed here, specifically so the two files can't drift onto
  different hyperparameters and produce backtest results that don't
  actually describe the deployed model.
- **"Win rate"** is defined as: of every BUY signal issued, what fraction
  had a positive actual forward return. This is a different (and looser)
  bar than "accuracy," which requires beating the +10% label threshold
  exactly — a BUY that returned +3% counts as a "win" here but not as a
  correct prediction in the accuracy sense. Both numbers are printed
  because they answer different questions ("was the model right about
  beating +10%?" vs. "would you have made money?").
- **Strategy returns per period** = the *average* forward return across
  every ticker that got a BUY signal on a given snapshot date, or 0.0
  ("sitting in cash") on any date with zero BUY signals.
- **Two different Sharpe/drawdown numbers are printed on purpose, and this
  is the single most important limitation to be upfront about**:
  - The **"NAIVE"** numbers compound the quarterly strategy-return series
    directly. This is wrong in a way that matters: each period's "return"
    is actually a *12-month-forward* return, and quarterly snapshots
    overlap by 9 of those 12 months. Compounding overlapping windows
    sequentially double- and triple-counts the same underlying market
    months, which is why the naive Sharpe (2.17 in the current run) looks
    unrealistically strong.
  - The **"HONEST"** numbers instead take only the *first* snapshot's
    return from each calendar year — a non-overlapping series (6 points for
    a 5-year backtest) — before computing Sharpe, drawdown, and a
    compounded total return that's actually comparable to the S&P 500's
    real buy-and-hold return over the same span. The tradeoff: only 6 data
    points makes for a very wide-uncertainty Sharpe/drawdown estimate — a
    genuinely correct methodology applied to too little data to fully trust
    on its own. Worth saying both things in the same breath in an interview.
  - Even the "honest" total-return comparison isn't perfectly apples-to-
    apples with the S&P figure: the S&P return is measured Jan 1 → Dec 31
    each calendar year, while the strategy's "annual" point is anchored to
    whatever date the first quarterly sample of that year happens to fall
    on (e.g. a February or March date) and its own 12-month-forward window
    — close, not identical.
- **Year-by-year breakdown** exists specifically so a bad year isn't
  hidden inside one aggregate accuracy number — e.g. the current run's 2021
  accuracy (0.300) is much worse than any other year, a real, explainable
  finding: 2021 samples predict 12 months forward into the 2022 rate-hike
  bear market, a regime shift the model's training data (a 2018–2021 mostly
  bull-market window at that point) had no examples of yet.

## `explain.py`

Turns a ticker's 5 raw scores into two sentences: `key_driver` (the highest
score — the strongest reason to like this stock) and `biggest_risk` (the
lowest score — the weakest link). This is intentionally a **simple
max/min over the ticker's own scores**, not a weighted combination with the
model's global feature importance — a factor can be globally important
across the training set on average while still not being what's actually
notable about *this specific ticker* (e.g. `risk_score` has the highest
average importance overall, but for a ticker with elite `quality_score` and
weak `valuation_score`, those are the more useful two sentences to surface).
Kept as a standalone module (not folded into `inference.py` or `app.py`) since
it's a distinct concern — presentation/explanation, not scoring or serving —
and pure/independently testable.

## `app.py`

Thin FastAPI wrapper around `inference.py` — intentionally has no ML/business
logic of its own.

- Keeping this file thin means the API layer can't silently diverge from
  what `inference.py` actually does, and makes it trivial to swap in a
  different web framework later without touching model-serving code.
- **CORS is wide open (`allow_origins=["*"]`)** — fine for a local demo
  where the frontend runs on a different origin (`localhost:5173` vs.
  `localhost:8000`); a real deployment would restrict this to the actual
  frontend origin.
- **The model loads once at startup**, not per-request — unpickling on
  every request would add latency for no benefit, since the model doesn't
  change between requests. If loading fails (model not trained yet), the
  server still starts (so `/` works) and fails informatively per-request in
  `/score` instead of crashing outright.
- **`/health` vs `/`**: `/` is a bare liveness ping (is the process up at
  all); `/health` additionally reports `model_loaded`, which is what you'd
  actually wire a real uptime/readiness check to — a server that's "up" but
  has no model loaded should read as unhealthy to a load balancer or
  monitoring tool, not as OK.
- **Feature importance is computed once at startup**, not per-request —
  it's a property of the trained model (constant across every ticker), so
  recomputing it on every `/score` call would be pure waste.
- **In-memory cache is a plain dict keyed by uppercased ticker**, storing
  `(timestamp, response)`. A request within `CACHE_TTL_SECONDS` (1 hour) of
  the last one for that ticker returns the stored response without
  re-fetching from `yfinance` or re-running inference. Deliberately not
  Redis or any external store — a single-process in-memory cache is the
  right amount of engineering for a local demo; it resets on restart and
  won't work across multiple server processes, both fine tradeoffs at this
  scale and explicitly flagged rather than silently assumed.
  - Only **successful** results are cached — an error (bad ticker, fetch
    failure) is never stored, so a transient failure doesn't get "stuck"
    for the full hour.
- **A global exception handler** (`@app.exception_handler(Exception)`)
  catches anything `inference.predict()` didn't already turn into a clean
  error, returning a JSON 500 instead of letting an unhandled exception
  produce FastAPI's default HTML error page or, worse, kill the worker.

## `frontend/`

React + Vite, plain CSS (no Tailwind — the existing custom stylesheet
already had a complete light/dark design system via CSS variables before
Phase 5 started, and rebuilding it in Tailwind would have thrown that away
for no functional gain).

- **`api.js` is the only file that knows the backend's URL/response shape.**
  `App.jsx` only ever deals with the already-parsed result object or a
  thrown `Error` with a human-readable `.message` — it has no fetch/JSON
  logic of its own, mirroring the same "thin caller, one file owns the
  contract" shape as `app.py` around `inference.py`.
- **The 5-score bar chart colors every bar green (≥0.5) or red (<0.5)**,
  not one fixed color for all five — this only works cleanly *because*
  `risk_score` was deliberately defined as "higher = lower risk" in
  `scoring/risk.py`; all five scores share the same "higher is more
  favorable to BUY" direction, so a single green/red threshold is
  meaningful for every bar without a special case for risk.
- **Chart colors reference CSS custom properties directly as SVG `fill`
  values** (`var(--buy)`, `var(--sell)`) rather than hardcoded hex codes,
  so the chart repaints correctly for both the light and dark palettes
  already defined in `index.css` without any chart-specific dark-mode code.
- **The explanation section reads directly from the API's `explanation`
  key** rather than recomputing key-driver/biggest-risk logic in the
  frontend — that logic lives in exactly one place (`explain.py`), so the
  UI can never show a different "biggest risk" than what the backend
  actually reasoned about.
- **Every new piece of the result card (`scores`, `explanation`) is
  guarded with `result.scores &&` / `result.explanation &&`** rather than
  assumed present — a defensive habit carried over from `inference.py`
  favoring "degrade gracefully" over "assume the shape."
