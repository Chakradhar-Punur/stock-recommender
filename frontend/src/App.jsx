import { useState } from "react";
import { fetchRecommendation } from "./api";
import "./App.css";

const EXAMPLE_TICKERS = ["MSFT", "AAPL", "GOOGL"];

/**
 * The whole app is one screen: a ticker input, an Analyze button, and a
 * result area that's always in exactly one of four states — idle,
 * loading, error, or showing a result. Small enough that splitting into
 * more components would just add indirection without real benefit.
 */
function App() {
  const [ticker, setTicker] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  async function analyze(symbolOverride) {
    const symbol = (symbolOverride ?? ticker).trim().toUpperCase();
    if (!symbol || isLoading) return;

    setTicker(symbol);
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await fetchRecommendation(symbol);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  }

  function handleSubmit(e) {
    e.preventDefault();
    analyze();
  }

  return (
    <div className="page">
      <div className="card">
        <header className="header">
          <div className="logo">📈</div>
          <h1>Stock Recommender</h1>
          <p className="subtitle">ML-powered BUY / SELL signal, scored live</p>
        </header>

        <form className="analyze-form" onSubmit={handleSubmit}>
          <input
            type="text"
            value={ticker}
            onChange={(e) => setTicker(e.target.value)}
            placeholder="Enter a ticker, e.g. MSFT"
            autoCapitalize="characters"
            autoCorrect="off"
            spellCheck="false"
          />
          <button type="submit" disabled={!ticker.trim() || isLoading}>
            {isLoading ? "Analyzing…" : "Analyze"}
          </button>
        </form>

        <div className="examples">
          <span>Try:</span>
          {EXAMPLE_TICKERS.map((t) => (
            <button
              key={t}
              type="button"
              className="chip"
              onClick={() => analyze(t)}
              disabled={isLoading}
            >
              {t}
            </button>
          ))}
        </div>

        <div className="result-area">
          {isLoading && (
            <div className="status loading">
              <span className="spinner" aria-hidden="true" />
              Analyzing {ticker.toUpperCase()}…
            </div>
          )}

          {error && (
            <div className="status error">
              <span aria-hidden="true">⚠️</span>
              {error}
            </div>
          )}

          {result && <ResultCard result={result} />}
        </div>

        <footer className="footer">Not investment advice — a portfolio ML demo</footer>
      </div>
    </div>
  );
}

/**
 * The color-coded output card — green for BUY, gray for HOLD, red for
 * SELL, with a confidence meter bar. (In practice the current model only
 * ever returns BUY/SELL, never HOLD, but the gray fallback stays here in
 * case that changes later.)
 */
function ResultCard({ result }) {
  const tone =
    result.recommendation === "BUY" ? "buy" : result.recommendation === "SELL" ? "sell" : "hold";
  const arrow = tone === "buy" ? "▲" : tone === "sell" ? "▼" : "—";

  return (
    <div className={`result-card ${tone}`}>
      <div className="result-top">
        <span className="ticker">{result.ticker}</span>
        <span className="recommendation">
          <span className="arrow">{arrow}</span>
          {result.recommendation}
        </span>
      </div>

      <div className="confidence-row">
        <div className="confidence-bar">
          <div className="confidence-fill" style={{ width: `${result.confidence}%` }} />
        </div>
        <span className="confidence-label">{result.confidence.toFixed(1)}%</span>
      </div>
    </div>
  );
}

export default App;
