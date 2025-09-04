import pandas as pd
from sklearn.linear_model import QuantileRegressor


class StopModel:
    """
    Predicts the (quantile) of MAE percentage (mae_pct) using features.
    This is a light, robust baseline with sklearn's QuantileRegressor.
    """
    def __init__(self, quantile: float = 0.9):
        self.model = QuantileRegressor(quantile=quantile, solver="highs")
        self.fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        self.fitted = True

    def predict(self, X: pd.DataFrame):
        if not self.fitted:
            raise ValueError("StopModel must be fitted before predicting.")
        return self.model.predict(X)
