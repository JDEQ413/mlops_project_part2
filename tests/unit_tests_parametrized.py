import pandas as pd
import pytest

import tests.funciones as tf


# Prueba para validar si existen datos para entrenamiento
def obtain_data_meta():
    return [("./mlops_project/data/", "data.csv")]


@pytest.mark.parametrize('path, filename', obtain_data_meta())
def testpy_data_exists(path, filename):
    assert tf.data_exists(path, filename) is True


# Prueba para validar Custom Transformer 1
def generate_dummies_transform_missingindicator():
    data = [[1.1, 1.7, 0.0, None, 5.08], [2.4, None, 0.1, 10.5, 6.09]]
    df = pd.DataFrame(data, columns=['C1', 'C2', 'C3', 'C4', 'C5'], index=None)
    return df


@pytest.mark.my_mark()
def testpy_custom_transformer_missing_indicator():
    test_data = generate_dummies_transform_missingindicator()
    assert tf.test_custom_transformer_missingindicator(test_data) is True


# Prueba para validar Custom Transformer 2
# --


# Prueba para validar si se guardó el modelo entrenado
def obtain_trained_model_meta():
    return [("./mlops_project/models/", "random_forest_output.pkl")]


@pytest.mark.parametrize('path, filename', obtain_trained_model_meta())
def testpy_trained_model_exists(path, filename):
    assert tf.trained_model_exist(path, filename) is True, False
