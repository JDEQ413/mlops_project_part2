"""Tests for 'mlops_project'."""

import os

import pandas as pd
import pytest

# from mlops_project.load.load_data import DataRetriever
from mlops_project.preprocess.preprocess_data import MissingIndicator

# from mlops_project.train.train_data import HousepricingDataPipeline


def test_csv_file_existence():
    """
    Test case to check if the data csv file exists.
    """

    MAIN_DIR = './mlops_project/'
    DATASETS_DIR = MAIN_DIR + 'data/'
    DATA_RETRIEVED = 'data.csv'

    file_exists = os.path.isfile(DATASETS_DIR + DATA_RETRIEVED)

    assert file_exists is True, f"The CSV file at '{(DATASETS_DIR + DATA_RETRIEVED)}' does not exist."


def test_missing_indicator_transform():
    """
    Test the 'transform' method of the MissingIndicator transformer.

    This test checks if the transformer correctly adds indicator features for missing values
    in the specified variables and returns the modified DataFrame.

    The test case uses a sample DataFrame with missing values and a custom transformer instance.

    It checks if the transformer successfully adds indicator features for the specified variables,
    and the transformed DataFrame has the expected additional columns.

    Note: Make sure to replace 'your_module' with the actual module name where the MissingIndicator class is defined.
    """
    # Sample DataFrame with missing values
    data = [[1.1, 1.7, 0.0, None, 5.08], [2.4, None, 0.1, 10.5, 6.09]]
    df = pd.DataFrame(data, columns=['C1', 'C2', 'C3', 'C4', 'C5'], index=None)
    missing_indicator = MissingIndicator(variables=['C1', 'C2', 'C3', 'C4', 'C5'])
    df_transformed = missing_indicator.transform(df)

    # Checks if dataframe contains missing indicator columns
    expected_columns = ['C1_nan', 'C2_nan', 'C3_nan', 'C4_nan', 'C5_nan']
    assert all(col in df_transformed.columns for col in expected_columns), \
        f"The dataframe after transformer must include the following additional columns: {expected_columns}"

    # Check if the missing indicator values are correct
    expected_values = [0, 0, 0, 1, 0]
    assert all(df_transformed[expected_columns].head(1) == expected_values), \
        f"Expected values for first sample: {expected_values}"

    expected_values = [0, 1, 0, 0, 0]
    assert all(df_transformed[expected_columns].tail(1) == expected_values), \
        f"Expected values for second sample: {expected_values}"

    # Check if the original DataFrame is not modified
    assert 'C1_nan' not in df.columns, "The original DataFrame should not be modified."


def test_missing_indicator_fit():
    """
    Test the 'fit' method of the MissingIndicator transformer.

    This test checks if the `fit` method returns the transformer instance itself,
    without performing any actual training or fitting.

    The test case uses a sample DataFrame and a custom transformer instance.

    Note: Make sure to replace 'your_module' with the actual module name where the MissingIndicator class is defined.
    """
    # Sample DataFrame
    data = [[1.1, 1.7, 0.0, None, 5.08], [2.4, None, 0.1, 10.5, 6.09]]
    df = pd.DataFrame(data, columns=['C1', 'C2', 'C3', 'C4', 'C5'], index=None)

    # Instantiate the custom transformer without specifying variables
    missing_indicator = MissingIndicator()

    # Fit the transformer to the DataFrame
    transformer_instance = missing_indicator.fit(df)

    # Check if the fit method returns the transformer instance itself
    assert transformer_instance == missing_indicator, \
        "The 'fit' method should return the transformer instance itself."


def test_trained_model_existence():
    """
    Test case to check if the trained model exists.
    """

    MAIN_DIR = './mlops_project/'
    DATASETS_DIR = MAIN_DIR + 'models/'
    DATA_RETRIEVED = 'random_forest_output.pkl'

    file_exists = os.path.isfile(DATASETS_DIR + DATA_RETRIEVED)

    assert file_exists is True, f"The trained model at '{(DATASETS_DIR + DATA_RETRIEVED)}' does not exist."


if __name__ == "__main__":
    # Run the test function using Pytest
    pytest.main([__file__])
