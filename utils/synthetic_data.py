import pandas as pd
import numpy as np


def generate_synthetic_bars(n: int = 2000, freq: str = "T") -> pd.DataFrame:
    """
    Create synthetic OHLCV bars with a random-walk 'close'.
    Index = RangeIndex(0..n-1). Includes open/high/low/close/volume columns.
    """
    rng = np.random.default_rng(42)
    # random-walk close
    steps = rng.normal(loc=0.0, scale=0.0008, size=n)   # ~8 bps per bar
    close = 4000 * np.cumprod(1 + steps)                 # around ES level
    # make OHLC around close
    spread = rng.uniform(0.25, 1.25, size=n)             # in index points
    open_ = close * (1 + rng.normal(0, 0.0003, size=n))
    high = np.maximum(open_, close) + spread * 0.5
    low = np.minimum(open_, close) - spread * 0.5
    volume = rng.integers(100, 5000, size=n)

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    })
    return df.reset_index(drop=True)


def generate_synthetic_signals(m: int, n_bars: int, horizon: int = 20) -> pd.DataFrame:
    """
    Produce 'm' signals with fields:
      - bar_index: integer in [0, n_bars - horizon - 1]
      - direction: +1 (long) or -1 (short)
    """
    if n_bars <= horizon + 1:
        raise ValueError("n_bars must be > horizon + 1 to place signals.")
    rng = np.random.default_rng(7)
    # Spread signals across the series with enough lookahead
    candidates = np.arange(0, n_bars - horizon - 1)
    bar_idx = rng.choice(candidates, size=min(m, len(candidates)), replace=False)
    direction = rng.choice([1, -1], size=len(bar_idx), replace=True)
    return pd.DataFrame({"bar_index": bar_idx, "direction": direction}).reset_index(drop=True)
