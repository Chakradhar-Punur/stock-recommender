import { useState } from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";
import { fetchRecommendation } from "./api";
import "./App.css";

const EXAMPLE_TICKERS = ["MSFT", "AAPL", "GOOGL"];

const SCORE_LABELS = {
  valuation_score: "Valuation",
  growth_score: "Growth",
  momentum_score: "Momentum",
  quality_score: "Quality",
  risk_score: "Risk",
};

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

      {result.scores && <ScoresChart scores={result.scores} />}

      {result.explanation && (
        <div className="explanation">
          <div className="explanation-row">
            <span className="explanation-label">Key driver</span>
            <span className="explanation-text">{result.explanation.key_driver}</span>
          </div>
          <div className="explanation-row">
            <span className="explanation-label">Biggest risk</span>
            <span className="explanation-text">{result.explanation.biggest_risk}</span>
          </div>
        </div>
      )}
    </div>
  );
}

function ScoresChart({ scores }) {
  const data = Object.entries(scores).map(([key, value]) => ({
    name: SCORE_LABELS[key] ?? key,
    value,
  }));

  return (
    <div className="scores-chart">
      <ResponsiveContainer width="100%" height={160}>
        <BarChart data={data} margin={{ top: 8, right: 4, left: -20, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" />
          <XAxis
            dataKey="name"
            tick={{ fontSize: 11, fill: "var(--text-muted)" }}
            axisLine={{ stroke: "var(--border)" }}
            tickLine={false}
          />
          <YAxis
            domain={[0, 1]}
            tick={{ fontSize: 11, fill: "var(--text-muted)" }}
            axisLine={false}
            tickLine={false}
            width={30}
          />
          <Tooltip
            formatter={(value) => value.toFixed(2)}
            contentStyle={{
              background: "var(--card-bg)",
              border: "1px solid var(--border)",
              borderRadius: 8,
              fontSize: 12,
            }}
          />
          <Bar dataKey="value" radius={[4, 4, 0, 0]}>
            {data.map((entry) => (
              <Cell key={entry.name} fill={entry.value >= 0.5 ? "var(--buy)" : "var(--sell)"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export default App;
