import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, features=None, scaler=False):
        self.features = features
        self.scaler = StandardScaler() if scaler else None

    def fit(self, X, y=None):
        if self.scaler:
            self.scaler.fit(X[self.features])
        return self

    def transform(self, X):
        X_transformed = X[self.features].copy()
        if self.scaler:
            X_transformed = self.scaler.transform(X_transformed)
        return X_transformed

class RegressionModel:
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor

    def train(self, X, y):
        X_preprocessed = self.preprocessor.fit_transform(X, y)
        self.model.fit(X_preprocessed, y)
        return self

    def predict(self, X):
        X_preprocessed = self.preprocessor.transform(X)
        return self.model.predict(X_preprocessed)
