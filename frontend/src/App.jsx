import { useState } from "react";
import { fetchRecommendation } from "./api";
import "./App.css";

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

  async function handleAnalyze(e) {
    e.preventDefault();
    const symbol = ticker.trim().toUpperCase();
    if (!symbol || isLoading) return;

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

  return (
    <div className="container">
      <h1>Stock Recommender</h1>
      <p className="hint">Try: MSFT, AAPL, GOOGL</p>

      <form className="analyze-form" onSubmit={handleAnalyze}>
        <input
          type="text"
          value={ticker}
          onChange={(e) => setTicker(e.target.value)}
          placeholder="Enter a ticker, e.g. MSFT"
          autoCapitalize="characters"
          autoCorrect="off"
        />
        <button type="submit" disabled={!ticker.trim() || isLoading}>
          Analyze
        </button>
      </form>

      {isLoading && <p className="status loading">Analyzing {ticker.toUpperCase()}...</p>}
      {error && <p className="status error">{error}</p>}
      {result && <ResultCard result={result} />}
    </div>
  );
}

/**
 * The color-coded output card — green for BUY, gray for HOLD, red for
 * SELL. (In practice the current model only ever returns BUY/SELL, never
 * HOLD, but the gray fallback stays here in case that changes later.)
 */
function ResultCard({ result }) {
  const toneClass =
    result.recommendation === "BUY" ? "buy" : result.recommendation === "SELL" ? "sell" : "hold";

  return (
    <div className={`result-card ${toneClass}`}>
      <span className="ticker">{result.ticker}</span>
      <span className="recommendation">{result.recommendation}</span>
      <span className="confidence">Confidence: {result.confidence.toFixed(1)}%</span>
    </div>
  );
}

export default App;
