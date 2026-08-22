"""
train_model.py
---------------
Trains an XGBoost classifier on the labeled dataset from
build_training_data.py, using a TIME-BASED train/test split.

WHY A TIME-BASED SPLIT (not a random one): our rows are financial time
series samples. A stock's momentum/valuation in Q1 2020 and its label
(did it beat +10% by Q1 2021) are influenced by the macro conditions of
that whole COVID-era window. If we split randomly, some rows from that
window end up in "train" and others from the *same* window end up in
"test" — the model can then implicitly learn "what generally happened in
2020-2021" from train and get an inflated test score just because train
and test overlap in time, not because it learned a real predictive
pattern. That's lookahead/leakage bias in disguise. A proper backtest-style
evaluation trains only on the past and tests only on data that comes
strictly after it in time — the same constraint a live trading system
would face (you can never train on the future). So here we sort by date
and cut a single boundary date: everything before it is train, everything
at/after it is test. No shuffling, and no date's full cross-section of
tickers is split across the boundary.
"""

import os
import pickle

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier


FEATURE_COLUMNS = ["momentum_score", "valuation_score"]
TARGET_COLUMN = "label"
TRAIN_FRACTION = 0.8  # fraction of unique sample dates used for training


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load the training CSV and parse dates for chronological sorting."""
    df = pd.read_csv(csv_path)
    df["sample_date"] = pd.to_datetime(df["sample_date"])
    return df.sort_values("sample_date").reset_index(drop=True)


def time_based_split(df: pd.DataFrame, train_fraction: float = TRAIN_FRACTION):
    """
    Split chronologically by a single cutoff DATE rather than by row count.

    Splitting by row count could cut a single sample_date's cross-section
    of ~10 tickers in half (some tickers land in train, others in test for
    the exact same date) — that still leaks same-period information across
    the split. Splitting by date boundary keeps every date's full
    cross-section on one side only.
    """
    unique_dates = sorted(df["sample_date"].unique())
    cutoff_idx = int(len(unique_dates) * train_fraction)
    cutoff_date = unique_dates[cutoff_idx]

    train_df = df[df["sample_date"] < cutoff_date]
    test_df = df[df["sample_date"] >= cutoff_date]
    return train_df, test_df, cutoff_date


def train_and_evaluate(df: pd.DataFrame):
    """
    Fit an XGBoost classifier on the chronological train split and
    evaluate it on the held-out (strictly later) test split.

    Returns (model, metrics_dict).
    """
    train_df, test_df, cutoff_date = time_based_split(df)

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COLUMN]

    print(f"Train/test split at cutoff date: {cutoff_date.date()}")
    print(f"  Train: {len(train_df)} rows "
          f"({train_df['sample_date'].min().date()} to {train_df['sample_date'].max().date()})")
    print(f"  Test:  {len(test_df)} rows "
          f"({test_df['sample_date'].min().date()} to {test_df['sample_date'].max().date()})")

    # Small, shallow model on purpose: only 2 features and ~230 training
    # rows in this first slice — a deep/high-capacity model would just
    # overfit noise. n_estimators/max_depth here are reasonable defaults
    # to revisit once more features and data are added.
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["HOLD/SELL (0)", "BUY (1)"])

    return model, {"accuracy": accuracy, "report": report}


def save_model(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    csv_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "processed", "training_data.csv")
    )
    model_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "models", "xgboost_v1.pkl")
    )

    if not os.path.exists(csv_path):
        print(f"[train_model] No training data found at {csv_path}. "
              f"Run build_training_data.py first.")
    else:
        dataset = load_dataset(csv_path)
        model, metrics = train_and_evaluate(dataset)

        print(f"\n=== Test set accuracy: {metrics['accuracy']:.3f} ===")
        print(f"\n=== Classification report ===\n{metrics['report']}")

        save_model(model, model_path)
        print(f"Model saved to {model_path}")
