import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from preprocess.preprocess_data import (DropMissing, IQR_DropOutliers,
                                        MissingIndicator, Standard_Scaler)
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


class HousepricingDataPipeline:
    """
    A class representing the House-price data processing and modeling pipeline.

    Attributes:
        features (list): A list of numerical variables in the dataset.
        target (str): A string with the target feature name.
        n (int): Number of outliers permited on features.
        seed_split (int): Seed value to reproduce results.
        model (RandomForest model): RandomForest Model trained.
        X_train (DataFrame): Dataframe with features values used to train the model, and obtain evaluation metrics.
        y_train (DataFrame): Dataframe with target values used to train the model, and obtain evaluation metrics.
        X_test (DataFrame): Dataframe with features values used to test the trained model, and obtain evaluation metrics.
        y_test (DataFrame): Dataframe with target values used to test the trained model, and obtain evaluation metrics.
        y_pred (lits or array-like): Data with predictions obtained from the testing the trained model.
        trained_model_dir (str): String with the path to store the trained model.
        file_save_name (str): String with the name of the model to be stored.
        scores_df (DataFrame): Dataframe with evaluation metrics.

    Methods:
        create_pipeline(): Creates and returns the House-pricing data processing pipeline.
        fit_random_forest(): Creates Random Forest model fit and returns it.
    """

    def __init__(self, features, target, n, seed_model):
        self.FEATURES = features
        self.TARGET = target
        self.n = n
        self.SEED_MODEL = seed_model

    def create_pipeline(self):
        """
        Creates and returns the House-pricing data processing pipeline.

        Returns:
            Pipeline: A scikit-learn pipeline for data processing and modeling.
        """
        self.PIPELINE = Pipeline(
            [
                ('missing_indicator', MissingIndicator(variables=self.FEATURES)),
                ('iqr_dropoutliers', IQR_DropOutliers(features=self.FEATURES, n=self.n)),
                ('drop_missing', DropMissing()),
                # ('oh_encoder', OneHotEncoder(features=CATEGORICAL_FEATURES))
                ('scaler', Standard_Scaler(features=self.FEATURES, target=self.TARGET))
                # ('scaler', StandardScaler())
            ]
        )
        return self.PIPELINE

    def fit_random_forest(self, X_train, y_train):
        """
        Fit a Random Forest model using the predefined data preprocessing pipeline.

        Parameters:
            X_train (pandas.DataFrame or numpy.ndarray): The training input data.
            y_train (pandas.Series or numpy.ndarray): The target values for training.

        Returns:
            random_forest (RandomForest model): The fitted Rangom Forest model.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.model = RandomForestRegressor(n_estimators=100, random_state=self.SEED_MODEL)
        self.model.fit(self.X_train, self.y_train)
        return self.model

    def predict(self, X_test):
        self.X_test = X_test
        self.y_pred = self.model.predict(self.X_test)
        return self.y_pred

    def get_evaluation_metrics(self, y_test):
        self.y_test = y_test
        cv_score = cross_val_score(estimator=self.model, X=self.X_train, y=self.y_train, cv=10)

        # Calculating Adjusted R-squared
        r2 = self.model.score(self.X_test, self.y_test)
        # Number of observations is the shape along axis 0
        n = self.X_test.shape[0]
        # Number of features (predictors, p) is the shape along axis 1
        p = self.X_test.shape[1]
        # Adjusted R-squared formula
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        RMSE = np.sqrt(metrics.mean_squared_error(self.y_test, self.y_pred))
        R2 = self.model.score(self.X_test, self.y_test)
        CV_R2 = cv_score.mean()

        self.scores_df = pd.DataFrame(data=[[R2, adjusted_r2, CV_R2, RMSE]], columns=['R2 Score', 'Adjusted R2 Score', 'Cross Validated R2 Score', 'RMSE'])
        self.scores_df.insert(0, 'Model', 'Random Forest')

        return self.scores_df

    def persist_model(self, trained_model_dir, file_save_name):
        # Saves the model recently trained

        if not os.path.isdir(trained_model_dir):   # Searches for the default models folder
            os.mkdir(Path(trained_model_dir))
        if os.path.isdir(trained_model_dir):
            joblib.dump(self.model, trained_model_dir + file_save_name)

        print()
        print("Model stored in: " + trained_model_dir + file_save_name)

    def make_prediction(self, X_values, selected_features, features=None):
        """
        Makes predictions with the model trained with values provided

        Parameters:
            X_values (pd.DataFrame):
            features (list):
            selected_features (list):
        """
        if features is not None:
            X_values.columns = features
        X_values = X_values[selected_features].copy()

        print(self.model.predict(X_values))
