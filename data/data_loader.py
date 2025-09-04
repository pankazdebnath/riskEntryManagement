import pandas as pd
from typing import Optional
from utils.synthetic_data import generate_synthetic_bars, generate_synthetic_signals


def load_bars_df(path: Optional[str] = None, n: int = 2000, freq: str = "T") -> pd.DataFrame:
    """
    Load OHLCV bars.
    - If path is given, supports CSV or Parquet.
    - If path is None, generates synthetic OHLCV with index = RangeIndex [0..n-1].
    Expected columns for downstream: at least 'close' (open/high/low/volume are fine too).
    """
    if path is None:
        return generate_synthetic_bars(n=n, freq=freq)
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        raise ValueError("Unsupported bars file format. Use .csv or .parquet or path=None for synthetic.")
    # Ensure RangeIndex
    df = df.reset_index(drop=True)
    if "close" not in df.columns:
        raise ValueError("bars_df must contain a 'close' column.")
    return df


def load_signals_df(
    path: Optional[str] = None,
    bars_df: Optional[pd.DataFrame] = None,
    m: int = 200,
    horizon: int = 20
) -> pd.DataFrame:
    """
    Load signals:
    - If path is given, supports CSV or Parquet; requires columns ['bar_index','direction'].
    - If path is None, generate synthetic signals aligned to bars_df.
    Notes:
      * bar_index must be < len(bars_df) - horizon
      * direction in {1, -1}
    """
    if path is None:
        if bars_df is None:
            raise ValueError("bars_df required when generating synthetic signals.")
        return generate_synthetic_signals(m=m, n_bars=len(bars_df), horizon=horizon)

    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        raise ValueError("Unsupported signals file format. Use .csv or .parquet or path=None for synthetic.")

    required = {"bar_index", "direction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"signals_df missing required columns: {missing}")
    df["bar_index"] = df["bar_index"].astype(int)
    return df.reset_index(drop=True)
