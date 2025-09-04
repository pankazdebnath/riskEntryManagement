import pandas as pd


def generate_features(bars_df):
    """Example technical indicators as features."""
    df = bars_df.copy()
    df["return_1"] = df["close"].pct_change()
    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_10"] = df["close"].rolling(10).mean()
    df["volatility"] = df["return_1"].rolling(10).std()
    return df.dropna()
