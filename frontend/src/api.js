/**
 * Talks to the FastAPI backend (src/app.py in the project root) — the
 * same /score/{ticker} endpoint, regardless of which frontend calls it.
 * Kept in its own file so App.jsx only deals with UI state, not fetch
 * details or error-shape parsing.
 */

const BASE_URL = "http://localhost:8000";

/**
 * Fetch a BUY/SELL recommendation for a ticker.
 *
 * @param {string} ticker
 * @returns {Promise<{ticker: string, recommendation: string, confidence: number}>}
 * @throws {Error} with a user-friendly message on any failure —
 *   network error (backend not running), or a real HTTP error from the
 *   API (bad ticker -> 404, model not trained -> 503). FastAPI's default
 *   error body is {"detail": "..."}, which we surface directly.
 */
export async function fetchRecommendation(ticker) {
  let response;
  try {
    response = await fetch(`${BASE_URL}/score/${ticker}`);
  } catch (networkError) {
    // fetch() itself throws for "can't reach the server at all" —
    // wrong port, backend not running, CORS blocked, etc.
    throw new Error("Can't reach the API. Is the backend running on localhost:8000?");
  }

  const body = await response.json();

  if (!response.ok) {
    // FastAPI's HTTPException body shape: {"detail": "..."}
    throw new Error(body.detail || `Server returned status ${response.status}.`);
  }

  return body;
}
