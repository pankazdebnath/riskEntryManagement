import pandas as pd


def generate_labels(bars_df, signals_df, horizon=5):
    """Generate stop and entry labels."""
    df = bars_df.join(signals_df, how="inner")
    future_max = df["close"].shift(-horizon).rolling(horizon).max()
    future_min = df["close"].shift(-horizon).rolling(horizon).min()
    df["stop_dist"] = df["close"] - future_min
    df["profit_dist"] = future_max - df["close"]
    df["label_entry"] = (df["profit_dist"] > df["stop_dist"]).astype(int)
    return df.dropna()
