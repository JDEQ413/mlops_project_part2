# Imports
import os
import shutil
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import opendatasets as od
import pandas as pd
from sklearn import metrics
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# CONSTANTS

DATASETS_DIR = './datasets/'
KAGGLE_URL = "https://www.kaggle.com/datasets/fedesoriano/the-boston-houseprice-data"
KAGGLE_LOCAL_DIR = KAGGLE_URL.split('/')[-1]
DATA_RETRIEVED = 'data.csv'

COLUMNS = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
TARGET = 'MEDV'
FEATURES = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
NUMERICAL_FEATURES = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'TAX', 'PTRATIO', 'B', 'LSTAT']
CATEGORICAL_FEATURES = ['CHAS', 'RAD']
SELECTED_FEATURES = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

YEO_JOHNSON_FEATURES = ['B']

SEED_SPLIT = 42
SEED_MODEL = 102

TRAINED_MODEL_DIR = 'trained_models/'
PIPELINE_NAME = 'random_forest'
PIPELINE_SAVE_FILE = f'{PIPELINE_NAME}_output.pkl'

# VARIABLES

droped_rows_index_list = []


# FUNCTIONS

def retrieve_data():

    # Downloads dataset from kaggle with pre-defined structure (folder)
    od.download(KAGGLE_URL, force=True)

    # Finds the recently downloaded file
    paths = sorted(Path(KAGGLE_LOCAL_DIR).iterdir(), key=os.path.getmtime)
    path_new_file = str(paths[-1])
    name_new_file = str(path_new_file).split('\\')[-1]

    # If recently downloaded file already exists in root, delete it
    if os.path.isfile(path_new_file):
        print("Dataset downloaded: " + path_new_file)
    else:
        print("Something went wrong, dataset not downloades!")

    # Moves the file to root instead of downloaded folder
    if os.path.isfile(DATASETS_DIR + name_new_file):            # Searches for the new file downloaded
        os.remove(DATASETS_DIR + name_new_file)                 # ,and deletes it
    if os.path.isfile(DATASETS_DIR + DATA_RETRIEVED):           # Searches for any old file with FILE_NAME specified
        os.remove(DATASETS_DIR + DATA_RETRIEVED)                # ,and deletes it too
    os.rename(path_new_file, DATASETS_DIR + DATA_RETRIEVED)     # Finally, moves downloaded file to default datasets folder
    print("And stored in: " + DATASETS_DIR + DATA_RETRIEVED)
    shutil.rmtree(KAGGLE_LOCAL_DIR)                         # Deletes the folder where kaggle library downloaded dataset


def Reg_Models_Evaluation_Metrics(model, X_train, y_train, X_test, y_test, y_pred):
    cv_score = cross_val_score(estimator=model, X=X_train, y=y_train, cv=10)

    # Calculating Adjusted R-squared
    r2 = model.score(X_test, y_test)
    # Number of observations is the shape along axis 0
    n = X_test.shape[0]
    # Number of features (predictors, p) is the shape along axis 1
    p = X_test.shape[1]
    # Adjusted R-squared formula
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    R2 = model.score(X_test, y_test)
    CV_R2 = cv_score.mean()

    return [[R2, adjusted_r2, CV_R2, RMSE]]


def get_features(df):
    # Obtains features that will be an input for the model
    # Extracts features from final dataframe, excluding TARGET and CATEGORICAL_FEATURES that were transformed to OHE

    df = df.drop(CATEGORICAL_FEATURES).copy()
    df = df.drop(TARGET).copy()
    return df.columns


def persist_model():
    # Saves the model recently trained

    if not os.path.isdir(TRAINED_MODEL_DIR):   # Searches for the default models folder
        os.mkdir(Path(TRAINED_MODEL_DIR))
    if os.path.isdir(TRAINED_MODEL_DIR):
        joblib.dump(RF_model, TRAINED_MODEL_DIR + PIPELINE_SAVE_FILE)

    print("Model stored in: " + TRAINED_MODEL_DIR + PIPELINE_SAVE_FILE)


# CUSTOM TRANSFORMERS

class MissingIndicator(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer to create indicator features for missing values in specified variables.

    Parameters:
        variables (list or str, optional): List of column names (variables) to create indicator features for.
            If a single string is provided, it will be treated as a single variable. Default is None.

    Attributes:
        variables (list): List of column names (variables) to create indicator features for.

    Methods:
        fit(X, y=None):
            This method does not perform any actual training or fitting.
            It returns the transformer instance itself.

        transform(X):
            Creates indicator features for missing values in the specified variables and returns the modified DataFrame.

    Example usage:
    ```
    from sklearn.pipeline import Pipeline

    # Instantiate the custom transformer
    missing_indicator = MissingIndicator(variables=['age', 'income'])

    # Define the pipeline with the custom transformer
    pipeline = Pipeline([
        ('missing_indicator', missing_indicator),
        # Other pipeline steps...
    ])

    # Fit and transform the data using the pipeline
    X_transformed = pipeline.fit_transform(X)
    ```
    """
    def __init__(self, variables=None):
        """
        Initialize the MissingIndicator transformer.

        Parameters:
            variables (list or str, optional): List of column names (variables) to create indicator features for.
                If a single string is provided, it will be treated as a single variable. Default is None.
        """
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X):
        """
        This method does not perform any actual training or fitting, as indicator features are created based on data.
        It returns the transformer instance itself.

        Parameters:
            X (pd.DataFrame): Input data to be transformed. Not used in this method.
            y (pd.Series or np.array, optional): Target variable. Not used in this method.

        Returns:
            self (MissingIndicator): The transformer instance.
        """
        return self

    def transform(self, X):
        """
        Creates indicator features for missing values in the specified variables and returns the modified DataFrame.

        Parameters:
            X (pd.DataFrame): Input data to be transformed.

        Returns:
            X_transformed (pd.DataFrame): Transformed DataFrame with additional indicator features for missing values.
        """
        X = X.copy()
        for var in self.variables:
            X[f'{var}_nan'] = X[var].isnull().astype(int) * 1

        return X


class IQR_DropOutliers (BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer that takes a dataframe and list of features to return the same dataframe modified excluding outliers according to the Tukey IQR method.

    Parameters:
        n (int): Number of outliers permited on features.
        features (list of string): List of numeric features on which to obtain outliers.
            If a single string is provided, it will be treated as a single variable. Default is None.

    Attributes:
        features (list): List of features on which to obtain outliers.

    Methods:
        fit(X, y=None):
            This method does not perform any actual training or fitting.
            It returns the transformer instance itself.

        transform(X):
            Finds outliers
            Creates indicator features for missing values in the specified variables and returns the modified DataFrame.

    Example usage:
    ```
    import numpy as np
    import pandas as pd
    from sklearn.pipeline import Pipeline

    # Instantiate the custom transformer
    iqr_dropoutliers = IQR_DropOutliers(n=1, features=['sales', 'transactions'])

    # Define the pipeline with the custom transformer
    pipeline = Pipeline([
        ('iqr_dropoutliers', iqr_dropoutliers),
        # Other pipeline steps...
    ])

    # Fit and transform the data using the pipeline
    X_transformed = pipeline.fit_transform(X)
    ```
    """

    def __init__(self, features=None, n=0):
        """
        Initialize the IQR_DropOutliers transformer.

        Parameters:
            features (list of string): List of numeric features on which to obtain outliers.
                If a single string is provided, it will be treated as a single variable. Default is None.
        """
        if not isinstance(features, list):
            self.features = [features]
        else:
            self.features = features

        self.n = n

    def fit(self, X, y=None):
        """
        This method does not perform any actual training or fitting.
        It returns the transformer instance itself.

        Parameters:
            X (pd.DataFrame): Input data to be transformed. Not used in this method.

        Returns:
            self (IQR_DropOutliers): The transformer instance.
        """
        return self

    def transform(self, X):
        """
        identifies outliers in the list of features to drop them from input dataframe and returns the modified dataframe.

        Parameters:
            X (pd.DataFrame): Input data to be transformed.

        Returns:
            X_transformed (pd.DataFrame): Transformed DataFrame without outliers.
        """

        outliers_index_list = []
        for feature in self.features:
            Q1 = np.percentile(X[feature], 25)      # 1st quartile (25%)
            Q3 = np.percentile(X[feature], 75)      # 3rd quartile (75%)
            IQR = Q3 - Q1                           # Interquartile range (IQR)
            outlier_step = 1.5 * IQR                # Outlier step for the current feature
            outlier_index_list = X[(X[feature] < Q1 - outlier_step) | (X[feature] > Q3 + outlier_step)].index   # Determining a list of indices of outliers
            outliers_index_list.extend(outlier_index_list)                                                      # appending the list of outliers

        # Selecting observations containing more than n outliers
        outlier_index_counts = Counter(outliers_index_list)
        outlieres_list = list(k for k, v in outlier_index_counts.items() if v > self.n)

        # Droping outliers found
        X = X.drop(outlieres_list, axis=0).reset_index(drop=True)

        return X


class DropMissing (BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer that takes missing indicator variables to drop records and drop "_na" variables later.

    Parameters:
        This transformer does not need parameters.

    Attributes:
        This transformer does not need attributes.

    Methods:
        fit(X):
            This method does not perform any actual training or fitting.
            It returns the transformer instance itself.

        transform(X):
            Drop NA records, and drop NA missing indicator variables later.
            Returns the modified DataFrame.

    Example usage:
    ```
    import numpy as np
    import pandas as pd
    from sklearn.pipeline import Pipeline

    # Instantiate the custom transformer
    dropna = DropMissing()

    # Define the pipeline with the custom transformer
    pipeline = Pipeline([
        ('dropna', DropMissing),
        # Other pipeline steps...
    ])

    # Fit and transform the data using the pipeline
    X_transformed = pipeline.fit_transform(X)
    ```
    """

    def __init__(self):
        """
        Initialize the DropMissing transformer.

        Parameters:
            This transformer does not need parameters.
        """

    def fit(self, X, y=None):
        """
        This method does not perform any actual training or fitting.
        It returns the transformer instance itself.

        Parameters:
            This transformer does not need parameters.

        Returns:
            self (DropMissing): The transformer instance.
        """
        return self

    def transform(self, X):
        """
        Based on missing values indicators identifies records with NA values to drop them from input dataframe and returns the modified dataframe.

        Parameters:
            This transformer does not need parameters.

        Returns:
            X_transformed (pd.DataFrame): Transformed DataFrame without missing values records and whitout missing indicator variables.
        """

        nans_index_list = []
        nan_columns = [col for col in X.columns if 'nan' in col]

        for column in nan_columns:
            nan_index_list = X[X[column] == 1].index   # Determining a list of indices of outliers
            nans_index_list.extend(nan_index_list)     # appending the list of outliers

        # Selecting observations containing missing values
        nan_index_counts = Counter(nans_index_list)
        self.nans_list = list(k for k, v in nan_index_counts.items() if v > 0)
        print(self.nans_list)

        # Droping records with missing found
        X = X.drop(self.nans_list, axis=0).reset_index(drop=True)
        X = X.drop(nan_columns, axis=1)               # Drops columns with postfix 'nan'

        return X


class OneHotEncoder(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer to perform one-hot encoding for categorical features.

    Parameters:
        features (list or str, optional): List of column names (features) to perform one-hot encoding for.
            If a single string is provided, it will be treated as a single variable. Default is None.

    Attributes:
        features (list): List of column names (features) to perform one-hot encoding for.
        dummies (list): List of column names representing the one-hot encoded dummy features.

    Methods:
        fit(X, y=None):
            Calculates the one-hot encoded dummy feature columns for the specified categorical features.
            It returns the transformer instance itself.

        transform(X):
            Performs one-hot encoding for the specified categorical features and returns the modified DataFrame.

    Example usage:
    ```
    from sklearn.pipeline import Pipeline

    # Instantiate the custom transformer
    ohe_encoder = OneHotEncoder(variables=['category1', 'category2'])

    # Define the pipeline with the custom transformer
    pipeline = Pipeline([
        ('ohe_encoder', ohe_encoder),
        # Other pipeline steps...
    ])

    # Fit and transform the data using the pipeline
    X_transformed = pipeline.fit_transform(X)
    ```
    """
    def __init__(self, features=None):
        """
        Initialize the OneHotEncoder transformer.

        Parameters:
            features (list or str, optional): List of column names (features) to perform one-hot encoding for.
                If a single string is provided, it will be treated as a single variable. Default is None.
        """
        self.features = [features] if not isinstance(features, list) else features

    def fit(self, X, y=None):
        """
        Calculates the one-hot encoded dummy variable columns for the specified categorical features.

        Parameters:
            X (pd.DataFrame): Input data to be transformed.

        Returns:
            self (OneHotEncoder): The transformer instance.
        """
        self.dummies = pd.get_dummies(X[self.features], drop_first=False).columns
        return self

    def transform(self, X):
        """
        Performs one-hot encoding for the specified categorical features and returns the modified DataFrame.

        Parameters:
            X (pd.DataFrame): Input data to be transformed.

        Returns:
            X_transformed (pd.DataFrame): Transformed DataFrame with one-hot encoded dummy variables for the specified categorical features.
        """
        X = X.copy()
        X = pd.concat([X, pd.get_dummies(X[self.features], drop_first=False)], axis=1)
        X.drop(self.features, axis=1)

        return X


class Standard_Scaler(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer to perform standard scaling on features except target feature.

    Parameters:
        features (list or str): List of column names (features) to perform StandardScaler.
            If a single string is provided, it will be treated as a single variable. Default is None.
            This list should not contain target feature.
        target (str): Name of the target feature which is not going to be transformed.

    Attributes:
        features (list): List of column names (features) to perform StandardScaler.
        target (str): Name of the target feature which is not going to be transformed.

    Methods:
        fit(X, y=None):
            This method does not perform any actual training or fitting.
            It returns the transformer instance itself.

        transform(X):
            Transforms features using StandardScaler, except on target feature.
            Returns the modified DataFrame.

    Example usage:
    ```
    from sklearn.pipeline import Pipeline

    # Instantiate the custom transformer
    standardscaler = Standard_Scaler(features=['category1', 'category2'], target='class')

    # Define the pipeline with the custom transformer
    pipeline = Pipeline([
        ('standardscaler', standardscaler),
        # Other pipeline steps...
    ])

    # Fit and transform the data using the pipeline
    X_transformed = pipeline.fit_transform(X)
    ```
    """
    def __init__(self, features=None, target=None):
        """
        Initialize the Standard_Scaler transformer.

        Parameters:
            features (list or str): List of column names (features) to perform StandardScaler.
                If a single string is provided, it will be treated as a single variable. Default is None.
                This list should not contain target feature.
            target (str): Name of the target feature which is not going to be transformed.
        """
        self.features = [features] if not isinstance(features, list) else features
        self.target = target

    def fit(self, X):
        """
        This method does not perform any actual training or fitting.
        It returns the transformer instance itself.

        Parameters:
            X (pd.DataFrame): Input data to be transformed.

        Returns:
            self (Standard_Scaler): The transformer instance.
        """
        return self

    def transform(self, X):
        """
        Performs Standard_Scaler for the specified features and returns the modified DataFrame.

        Parameters:
            X (pd.DataFrame): Input data to be transformed.

        Returns:
            X_transformed (pd.DataFrame): Transformed DataFrame.
        """

        self.standard_scaler = StandardScaler()
        X = X.copy()
        X_features_transformed = pd.DataFrame(self.standard_scaler.fit_transform(X[self.features]))
        X_features_transformed.columns = self.features
        X = pd.concat([X_features_transformed, X[self.target]], axis=1)

        return X


# PIPELINE

transform_pipeline = Pipeline(
    [
        ('missing_indicator', MissingIndicator(variables=FEATURES)),
        ('iqr_dropoutliers', IQR_DropOutliers(features=FEATURES, n=1)),
        ('drop_missing', DropMissing()),
        # ('oh_encoder', OneHotEncoder(features=CATEGORICAL_FEATURES))
        ('scaler', Standard_Scaler(features=FEATURES, target=TARGET))
        # ('scaler', StandardScaler())
    ]
)

# Load dataset
retrieve_data()
raw_df = pd.read_csv(DATASETS_DIR + DATA_RETRIEVED, delimiter=",")
df_transformed = transform_pipeline.fit_transform(raw_df)
X_train, X_test, y_train, y_test = train_test_split(df_transformed.drop(TARGET, axis=1), df_transformed[TARGET], test_size=0.2, random_state=SEED_SPLIT)

# Creating and training model
RF_model = RandomForestRegressor(n_estimators=100, random_state=SEED_MODEL)
RF_model.fit(X_train, y_train)

# Model making a prediction on test data
y_pred = RF_model.predict(X_test)
evaluation_metrics = Reg_Models_Evaluation_Metrics(RF_model, X_train, y_train, X_test, y_test, y_pred)

rf_score = pd.DataFrame(data=evaluation_metrics, columns=['R2 Score', 'Adjusted R2 Score', 'Cross Validated R2 Score', 'RMSE'])
rf_score.insert(0, 'Model', 'Random Forest')
print(rf_score)


# PERSISTINT THE TRAINED MODEL

# Save the model using joblib
persist_model()


# PREDICTIONS

new_data_pred = pd.DataFrame([[0.06905, 0.0, 2.18, 0, 0.458, 7.147, 54.2, 6.0622, 3, 222.0, 18.7, 396.90, 5.33]], columns=FEATURES)
new_data_pred = new_data_pred[SELECTED_FEATURES].copy()
print(RF_model.predict(new_data_pred))

new_data_pred = pd.DataFrame([[0.02729, 0.0, 7.07, 0, 0.469, 7.185, 61.1, 4.9671, 2, 242.0, 17.8, 392.83, 4.03]], columns=FEATURES)
new_data_pred = new_data_pred[SELECTED_FEATURES].copy()
print(RF_model.predict(new_data_pred))

new_data = X_test.tail(10)
new_data = new_data[SELECTED_FEATURES].copy()
print(RF_model.predict(new_data))
print(y_test.tail(10))
