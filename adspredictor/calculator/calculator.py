# calculator.py

import joblib

class AdsEnergyPredictor:
    def __init__(self, pipeline_filename):
        """
        Initializes the Calculator class by loading a trained pipeline.

        Parameters:
        - pipeline_filename: str, path to the saved trained pipeline
        """
        self.pipeline = self.load_pipeline(pipeline_filename)

    def load_pipeline(self, filename):
        """Loads a trained pipeline from a file."""
        pipeline = joblib.load(filename)
        return pipeline

    def predict(self, X_new):
        """
        Uses the loaded pipeline to make predictions on new data.

        Parameters:
        - X_new: pandas DataFrame or numpy array of new features

        Returns:
        - predictions: array of predicted values
        """
        predictions = self.pipeline.predict(X_new)
        return predictions