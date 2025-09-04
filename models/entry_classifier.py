from sklearn.ensemble import RandomForestClassifier


class EntryClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
