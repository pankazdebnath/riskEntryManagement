import pandas as pd
import numpy as np


def _forward_returns(close: pd.Series, idx: int, horizon: int) -> np.ndarray:
    """Return future simple returns for bars idx+1..idx+horizon relative to price at idx."""
    p0 = float(close.iloc[idx])
    future = close.iloc[idx + 1: idx + 1 + horizon].values
    return (future - p0) / p0


def generate_labels(bars_df: pd.DataFrame, signals_df: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    """
    Create labels per signal:
      - bar_index: entry bar index
      - direction: {1, -1}
      - ret_forward: simple return over horizon for the final bar (t+h)
      - mae_pct: approx. maximum adverse excursion (positive number, pct of price)
      - is_win: 1 if direction*ret_forward > 0 else 0
    Notes:
      * Uses close-to-close returns only (no intrabar highs/lows).
    """
    close = bars_df["close"].astype(float).reset_index(drop=True)
    n = len(close)

    rows = []
    for _, r in signals_df.iterrows():
        i = int(r["bar_index"])
        d = int(r["direction"])
        if i < 0 or i + horizon >= n:
            # skip signals that don't have enough lookahead
            continue

        fr = _forward_returns(close, i, horizon)
        # final horizon return
        ret_forward = (close.iloc[i + horizon] - close.iloc[i]) / close.iloc[i]
        # MAE approx: min adverse return over the horizon given the direction
        # For long: adverse move is negative returns (take min). For short: reverse sign.
        directional_returns = d * fr
        mae_pct = float(np.clip(-directional_returns.min(), a_min=0.0, a_max=None))  # positive

        rows.append({
            "bar_index": i,
            "direction": d,
            "ret_forward": float(ret_forward),
            "mae_pct": mae_pct,
            "is_win": 1 if d * ret_forward > 0 else 0,
        })

    return pd.DataFrame(rows).reset_index(drop=True)
