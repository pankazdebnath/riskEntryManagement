import pandas as pd
import numpy as np


def generate_sample_bars_df(n_rows: int = 500, start_date: str = "2024-01-01") -> pd.DataFrame:
    """
    Generate a sample OHLCV bars DataFrame.

    Parameters:
    -----------
    n_rows : int
        Number of rows (time steps) to generate.
    start_date : str
        Start date of the time series.

    Returns:
    --------
    pd.DataFrame with columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    """
    rng = pd.date_range(start=start_date, periods=n_rows, freq="H")
    prices = np.cumsum(np.random.randn(n_rows)) + 100  # random walk around 100
    high = prices + np.random.rand(n_rows) * 2
    low = prices - np.random.rand(n_rows) * 2
    open_ = prices + np.random.randn(n_rows) * 0.5
    close = prices + np.random.randn(n_rows) * 0.5
    volume = np.random.randint(100, 1000, size=n_rows)

    bars_df = pd.DataFrame({
        "timestamp": rng,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    })
    return bars_df


def generate_sample_signals_df(bars_df: pd.DataFrame, signal_prob: float = 0.05) -> pd.DataFrame:
    """
    Generate a sample signals DataFrame aligned with bars_df.

    Parameters:
    -----------
    bars_df : pd.DataFrame
        The OHLCV bars DataFrame to align signals with.
    signal_prob : float
        Probability of a signal occurring at each row.

    Returns:
    --------
    pd.DataFrame with columns: ['timestamp', 'signal']
    """
    rng = bars_df["timestamp"]
    signals = np.random.choice([0, 1], size=len(rng), p=[1 - signal_prob, signal_prob])

    signals_df = pd.DataFrame({
        "timestamp": rng,
        "signal": signals
    })
    return signals_df


# Example usage
if __name__ == "__main__":
    bars_df = generate_sample_bars_df(200)
    signals_df = generate_sample_signals_df(bars_df, signal_prob=0.1)

    print(bars_df)
    print(signals_df)
