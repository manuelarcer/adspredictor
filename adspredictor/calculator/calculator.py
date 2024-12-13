# calculator.py

import joblib
import os
from typing import Optional
#from sklearn.pipeline import Pipeline
from ..trainer.trainer import Trainer

class AdsEnergyPredictor:
    def __init__(
        self, 
        pipeline: Optional[Trainer] = None, 
        pipeline_filename: Optional[str] = None
    ):
        """
        Initializes the AdsEnergyPredictor either with a trained pipeline object or by loading 
        a pipeline from a file.

        Parameters:
        - pipeline (Pipeline, optional): An already trained scikit-learn Pipeline object.
        - pipeline_filename (str, optional): Path to the saved trained pipeline file.

        Raises:
        - ValueError: If neither or both parameters are provided.
        - FileNotFoundError: If the provided filename does not exist.
        - Exception: If loading the pipeline fails.
        """
        if pipeline and pipeline_filename:
            raise ValueError("Provide either 'pipeline' or 'pipeline_filename', not both.")
        if not pipeline and not pipeline_filename:
            raise ValueError("You must provide either 'pipeline' or 'pipeline_filename'.")

        if pipeline:
            if not isinstance(pipeline, Trainer):
                raise TypeError(f"'pipeline' must be a scikit-learn Pipeline object, got {type(pipeline)} instead.")
            self.pipeline = pipeline
            print("Initialized AdsEnergyPredictor with a provided pipeline object.")
        else:
            if not isinstance(pipeline_filename, str):
                raise TypeError(f"'pipeline_filename' must be a string, got {type(pipeline_filename)} instead.")
            if not os.path.isfile(pipeline_filename):
                raise FileNotFoundError(f"The file '{pipeline_filename}' does not exist.")
            self.pipeline = self.load_pipeline(pipeline_filename)
            print(f"Initialized AdsEnergyPredictor by loading pipeline from '{pipeline_filename}'.")

    def load_pipeline(self, filename: str) -> Trainer:
        """
        Loads a trained pipeline from a file.

        Parameters:
        - filename (str): Path to the saved trained pipeline file.

        Returns:
        - Pipeline: The loaded scikit-learn Pipeline object.

        Raises:
        - Exception: If loading the pipeline fails.
        """
        try:
            pipeline = joblib.load(filename)
            if not isinstance(pipeline, Trainer):
                raise TypeError(f"The loaded object is not a scikit-learn Pipeline, got {type(pipeline)} instead.")
            return pipeline
        except Exception as e:
            raise Exception(f"Failed to load the pipeline from '{filename}': {e}")

    def predict(self, X_new):
        """
        Uses the loaded pipeline to make predictions on new data.

        Parameters:
        - X_new: pandas DataFrame or numpy array of new features

        Returns:
        - predictions: array of predicted values
        """

        # The first pipeline step is to extract Trainer, the second is the scikit-learn model
        # predict only can be called on the scikit-learn model
        predictions = self.pipeline.pipeline.predict(X_new)
        return predictions