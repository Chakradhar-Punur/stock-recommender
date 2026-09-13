# Stock Recommender — Frontend

React (Vite) UI for the Stock Recommender API. See the
[root README](../README.md) for the full project, setup, and backtest
results.

## Run it

```bash
npm install
npm run dev
```

Needs the FastAPI backend running at `http://localhost:8000` (see the root
README) — `src/api.js` is the only file that knows that URL.

## Stack

React 19, Vite, recharts (5-score bar chart), plain CSS with light/dark
mode via CSS custom properties.
