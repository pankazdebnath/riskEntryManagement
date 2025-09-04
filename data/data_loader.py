import pandas as pd
import numpy as np


def load_bars_df():
    # Generate or load bars_df (OHLCV).
    dates = pd.date_range("2023-01-01", periods=500, freq="T")
    data = {
        "open": np.random.rand(len(dates)) * 100,
        "high": np.random.rand(len(dates)) * 100,
        "low": np.random.rand(len(dates)) * 100,
        "close": np.random.rand(len(dates)) * 100,
        "volume": np.random.randint(1, 1000, size=len(dates)),
    }
    return pd.DataFrame(data, index=dates)


def load_signals_df(bars_df):
    #   Generate or load signals_df aligned to bars_df.
    return pd.DataFrame({
        "entry_signal": np.random.choice([0, 1], size=len(bars_df))
    }, index=bars_df.index)

# bars = load_bars_df()
# signals = load_signals_df(bars)
# print(bars.head())
# print(signals.head())
