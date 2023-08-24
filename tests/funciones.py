import os

import pandas as pd

from server.preprocess.preprocess_data import MissingIndicator


def data_exists(path: str, filename: str) -> bool:
    if os.path.isfile(path + filename):
        return True
    else:
        return False


def test_custom_transformer_missingindicator(df: pd.DataFrame) -> bool:
    missing_indicator = MissingIndicator(variables=df.columns)
    df2 = missing_indicator.transform(df)
    print(df2)
    return True


def test_custom_transformer_dropmissing():
    pass


def trained_model_exist(path: str, filename: str) -> bool:
    if os.path.isfile(path + filename):
        return True
    else:
        return False
