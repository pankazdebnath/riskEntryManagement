from data.data_loader import load_bars_df, load_signals_df
from data.feature_engineering import engineer_features
from data import generate_labels
from models.stop_model import StopModel
from models.entry_model import EntryModel
from backtester.backtester import Backtester
from utils.metrics import print_metrics


def main():
    # ---------- CONFIG ----------
    horizon = 20           # forward horizon in bars for labels
    n_bars = 2000          # synthetic bar count when no file path is given
    n_signals = 250        # synthetic signals when no file path is given
    entry_threshold = 0.5  # probability cutoff for entry filter

    # ---------- LOAD DATA ----------
    bars_df = load_bars_df(path=None, n=n_bars, freq="T")  # path=None -> synthetic data
    signals_df = load_signals_df(path=None, bars_df=bars_df, m=n_signals, horizon=horizon)

    # ---------- FEATURES & LABELS ----------
    features_df = engineer_features(bars_df)                # creates columns: ret_1, volatility, sma_10, sma_50
    labels_df = generate_labels(bars_df, signals_df, horizon=horizon)
    # Keep only entries that exist after feature dropna
    labels_df = labels_df[labels_df["bar_index"].isin(features_df.index)]

    # ---------- MODELS ----------
    stop_model = StopModel(quantile=0.9)
    entry_model = EntryModel()

    # ---------- BACKTEST ----------
    backtester = Backtester(stop_model, entry_model, entry_threshold=entry_threshold)
    results_df, performance = backtester.run_backtest(features_df, labels_df)

    # ---------- OUTPUT ----------
    print_metrics(results_df, performance)


if __name__ == "__main__":
    main()
