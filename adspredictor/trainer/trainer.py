# trainer.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

class Trainer:
    def __init__(self, dataframe, target_column, model_class, model_params=None, test_size=0.2, random_state=None):
        """
        Initializes the Trainer class.

        Parameters:
        - dataframe: pandas DataFrame containing features and target
        - target_column: string, name of the target column in dataframe
        - model: scikit-learn estimator class (e.g., LinearRegression, SVR)
        - model_params: dict, parameters to initialize the model
        - test_size: float, proportion of the dataset to include in the test split
        - random_state: int, random state for reproducibility
        """
        self.dataframe = dataframe
        self.target_column = target_column
        self.model_class = model_class
        self.test_size = test_size
        self.random_state = random_state
        self.model_params = model_params if model_params else {}
        self.pipeline = self.initialize_pipeline()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def initialize_model(self):
        """
        Initializes the model with provided parameters.
        """
        return self.model_class(**self.model_params)
    
    def split_data(self):
        """
        Splits the data into training and test sets.
        """
        X = self.dataframe.drop(columns=[self.target_column])
        y = self.dataframe[self.target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state)
    
    def initialize_pipeline(self):
        """Initializes the pipeline with preprocessing and model."""
        
        # Initialize the model with parameters if any
        try:
            model = self.model_class(**self.model_params)
        except TypeError as e:
            raise TypeError(
                    f"Error initializing model: {e}. "
                    f"Check if model_params are compatible with {self.model_class.__name__}."
                    )
        
        # Create the pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        return pipeline
    
    def train(self):
        """
        Trains the pipeline on the training data.
        """
        if self.X_train is None or self.y_train is None:
            self.split_data()
        self.pipeline.fit(self.X_train, self.y_train)

    def save_pipeline(self, filename):
        """Saves the trained pipeline to a file."""
        joblib.dump(self.pipeline, filename)
    
    def evaluate(self):
        """
        Evaluates the model on the test data.

        Returns:
        - metrics: dict containing MAE, RMSE, and R^2 score
        """
        if self.X_test is None or self.y_test is None:
            self.split_data()
        y_pred = self.pipeline.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        r2 = r2_score(self.y_test, y_pred)
        metrics = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
        return metrics
    
    def cross_validate(self, cv=5, scoring='neg_mean_absolute_error'):
        """
        Performs cross-validation on the entire dataset.

        Parameters:
        - cv: int, number of cross-validation folds
        - scoring: str, scoring method

        Returns:
        - cv_scores: array of cross-validation scores
        """
        X = self.dataframe.drop(columns=[self.target_column])
        y = self.dataframe[self.target_column]
        cv_scores = cross_val_score(self.pipeline, X, y, cv=cv, scoring=scoring)
        return -cv_scores
    
    def grid_search(self, param_grid, cv=5, scoring='neg_mean_absolute_error'):
        """
        Performs grid search to find the best hyperparameters.

        Parameters:
        - param_grid: dict, parameter grid to search
        - cv: int, number of cross-validation folds
        - scoring: str, scoring method

        Updates:
        - self.model: best estimator found by grid search

        Returns:
        - best_params: dict, best parameters found
        - best_score: float, best score achieved
        """
        X = self.dataframe.drop(columns=[self.target_column])
        y = self.dataframe[self.target_column]
        grid_search = GridSearchCV(self.model_class(), param_grid, cv=cv, scoring=scoring)
        grid_search.fit(X, y)
        self.model = grid_search.best_estimator_
        return grid_search.best_params_, grid_search.best_score_
    
    def save_model(self, filename):
        """
        Saves the trained model to a file.

        Parameters:
        - filename: str, path to save the model
        """
        joblib.dump(self.model, filename)
    
    def load_model(self, filename):
        """
        Loads a trained model from a file.

        Parameters:
        - filename: str, path to load the model from
        """
        self.model = joblib.load(filename)
