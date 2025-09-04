import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


class QuantileStopModel:
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        self.models = {
            q: GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=100)
            for q in quantiles
        }
        self.quantiles = quantiles

    def fit(self, X, y):
        for q, model in self.models.items():
            model.fit(X, y)

    def predict(self, X):
        preds = {}
        for q, model in self.models.items():
            preds[q] = model.predict(X)
        return preds
