import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier


class EntryModel:
    """
    Calibrated-ish classifier baseline (probabilities from GBC) for entry quality.
    """
    def __init__(self, **kwargs):
        self.model = GradientBoostingClassifier(random_state=42, **kwargs)
        self.fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        self.fitted = True

    def predict_proba(self, X: pd.DataFrame):
        if not self.fitted:
            raise ValueError("EntryModel must be fitted before predicting.")
        # GBC has predict_proba
        return self.model.predict_proba(X)[:, 1]
