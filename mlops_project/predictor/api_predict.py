import argparse
import os
import sys

import joblib

# setting path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
os.chdir(parent_dir)

from utilities.custom_loging import CustomLogging

logger = CustomLogging()
logger = logger.Create_Logger(logger_name='api_predict.log')


class ModelAPIPredictor:
    """
    A class to load a trained machine learning model and make predictions on new data.

    Parameters:
        trained_model_path (str): Path to the trained model file (joblib format).

    Methods:
        predict(new_data):
            Makes predictions on the provided new_data using the loaded trained model.
    """

    def __init__(self, model_path):
        """
        Initializes the ModelPredictor instance.

        Parameters:
            model_path (str): Path to the trained model file (joblib format).
        """
        logger.debug("ModelAPIPredictor initiated.")
        self.model = joblib.load(model_path)

    def predict(self, new_data):
        """
        Makes predictions on the provided new_data using the loaded model.

        Parameters:
            new_data: The data on which to make predictions.

        Returns:
            Predicted outputs from the model.
        """
        logger.debug("Prediction was requested.")
        return self.model.predict(new_data)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Model Predictor')
    parser.add_argument('model_path', type=str, help='Path to the trained model file')
    parser.add_argument('new_data', type=str, help='Path to the file containing new data for prediction')
    args = parser.parse_args()

    predictor = ModelAPIPredictor(args.model_path)

    new_data = args.new_data

    predictions = predictor.predict(new_data)
    logger.debug("Prediction generated.")
    print(predictions)
