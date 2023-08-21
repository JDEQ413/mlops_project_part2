from collections import Counter

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


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
