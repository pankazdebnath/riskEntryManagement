import pandas as pd


def engineer_features(bars_df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal features for demonstration. Keeps a RangeIndex aligned to bars_df.
    Outputs columns:
      - ret_1: 1-step simple return
      - volatility: rolling std of ret_1 (window=20)
      - sma_10, sma_50: simple moving averages
    """
    df = bars_df.copy().reset_index(drop=True)

    close = df["close"].astype(float)
    df["ret_1"] = close.pct_change().fillna(0.0)
    df["volatility"] = df["ret_1"].rolling(20).std().fillna(0.0)
    df["sma_10"] = close.rolling(10).mean().fillna(method="bfill")
    df["sma_50"] = close.rolling(50).mean().fillna(method="bfill")

    # Keep only the feature columns (index = bar number)
    return df[["ret_1", "volatility", "sma_10", "sma_50"]]
