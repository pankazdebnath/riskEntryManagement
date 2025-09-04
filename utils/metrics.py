import numpy as np
from typing import Iterable


def win_rate(trade_returns: Iterable[float]) -> float:
    tr = np.array(list(trade_returns), dtype=float)
    if tr.size == 0:
        return 0.0
    return float(np.mean(tr > 0))


def profit_factor(trade_returns: Iterable[float]) -> float:
    tr = np.array(list(trade_returns), dtype=float)
    gp = tr[tr > 0].sum()
    gl = -tr[tr < 0].sum()
    if gl == 0:
        return float(np.inf if gp > 0 else 0.0)
    return float(gp / gl)


def avg_win_loss_ratio(trade_returns: Iterable[float]) -> float:
    tr = np.array(list(trade_returns), dtype=float)
    wins = tr[tr > 0]
    losses = -tr[tr < 0]
    if len(losses) == 0:
        return float(np.inf if len(wins) > 0 else 0.0)
    return float(wins.mean() / losses.mean()) if len(wins) else 0.0


def sharpe_ratio_on_trades(trade_returns: Iterable[float]) -> float:
    tr = np.array(list(trade_returns), dtype=float)
    if tr.size < 2:
        return 0.0
    mu, sd = tr.mean(), tr.std(ddof=1)
    return float(mu / (sd + 1e-12))


def print_metrics(results_df, performance: dict):
    print("=== Backtest Performance Summary ===")
    for k, v in performance.items():
        print(f"{k:28s}: {v}")
    print("\n=== Results sample ===")
    print(results_df.head(10))
