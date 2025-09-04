import pandas as pd
import numpy as np
from utils.metrics import (
    win_rate, profit_factor, avg_win_loss_ratio, sharpe_ratio_on_trades
)


class Backtester:
    """
    Simple backtest that:
      * Joins labels with features by bar_index
      * Trains EntryModel on is_win, StopModel on mae_pct
      * Applies entry probability filter (threshold)
      * Computes metrics on the filtered trades using ret_forward
    """
    def __init__(self, stop_model, entry_model, entry_threshold: float = 0.5):
        self.stop_model = stop_model
        self.entry_model = entry_model
        self.entry_threshold = entry_threshold

    def run_backtest(self, features_df: pd.DataFrame, labels_df: pd.DataFrame):
        # Join features by bar_index -> feature row index
        data = labels_df.merge(
            features_df,
            how="inner",
            left_on="bar_index",
            right_index=True,
            suffixes=("", "_feat")
        )

        # Features and targets
        feature_cols = ["ret_1", "volatility", "sma_10", "sma_50"]
        X = data[feature_cols].astype(float)
        y_entry = data["is_win"].astype(int)
        y_stop = data["mae_pct"].astype(float)

        # Fit models
        self.entry_model.fit(X, y_entry)
        self.stop_model.fit(X, y_stop)

        # Predictions
        entry_prob = self.entry_model.predict_proba(X)
        mae_q = self.stop_model.predict(X)  # predicted 90th-quantile adverse move (fraction)

        # Apply entry filter
        take = entry_prob >= self.entry_threshold
        realized = data.loc[take, "direction"].values * data.loc[take, "ret_forward"].values

        # ---- Metrics ----
        metrics = {
            "n_signals": int(len(data)),
            "n_trades": int(take.sum()),
            "entry_threshold": float(self.entry_threshold),
            "win_rate": win_rate(realized),
            "profit_factor": profit_factor(realized),
            "avg_win_loss_ratio": avg_win_loss_ratio(realized),
            "sharpe_on_trades": sharpe_ratio_on_trades(realized),
            "predicted_stop_mae90_median_pct": float(np.median(mae_q)) if len(mae_q) else np.nan,
        }

        # Output table
        out = data.copy()
        out["entry_prob"] = entry_prob
        out["mae_pred_q"] = mae_q
        out["take"] = take
        out["trade_ret"] = 0.0
        out.loc[take, "trade_ret"] = realized

        return out, metrics
