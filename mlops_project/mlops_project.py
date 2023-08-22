"""Main module."""

import pandas as pd
from load.load_data import DataRetriever
from sklearn.model_selection import train_test_split
from train.train_data import HousepricingDataPipeline
from utilities.custom_loging import CustomLogging

logger = CustomLogging()
logger = logger.Create_Logger(logger_name='mlops_project.log', flag_streamer=True)
logger.info("Logger started.")

MAIN_DIR = './mlops_project/'
DATASETS_DIR = MAIN_DIR + 'data/'
KAGGLE_URL = "fedesoriano/the-boston-houseprice-data"
KAGGLE_FILE = 'boston.csv'
DATA_RETRIEVED = 'data.csv'

COLUMNS = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
TARGET = 'MEDV'
FEATURES = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
NUMERICAL_FEATURES = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'TAX', 'PTRATIO', 'B', 'LSTAT']
CATEGORICAL_FEATURES = ['CHAS', 'RAD']
SELECTED_FEATURES = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

SEED_SPLIT = 42
SEED_MODEL = 102

TRAINED_MODEL_DIR = MAIN_DIR + 'models/'
PIPELINE_NAME = 'random_forest'
PIPELINE_SAVE_FILE = f'{PIPELINE_NAME}_output.pkl'

logger.info("Constants defined.")

# droped_rows_index_list = []

if __name__ == "__main__":

    # Retrieve data
    # data_retriever = DataRetriever([DATASETS_DIR, KAGGLE_URL, KAGGLE_LOCAL_DIR, DATA_RETRIEVED])
    data_retriever = DataRetriever([MAIN_DIR, DATASETS_DIR, KAGGLE_URL, KAGGLE_FILE, DATA_RETRIEVED])
    logger.debug("Data retriever instantiated.")
    result = data_retriever.retrieve_data()
    logger.debug("Data retrieved.")

    # Read data
    raw_df = data_retriever.load_data()
    logger.debug("Data loaded.")

    # House-pricing Pipeline
    housepricing_pipeline = HousepricingDataPipeline(features=FEATURES, target=TARGET, n=1, seed_model=SEED_MODEL)
    housepricing_pipeline.create_pipeline()
    df_transformed = housepricing_pipeline.PIPELINE.fit_transform(raw_df)
    X_train, X_test, y_train, y_test = train_test_split(df_transformed.drop(TARGET, axis=1), df_transformed[TARGET], test_size=0.2, random_state=SEED_SPLIT)
    logger.debug("Pipeline executed. Dataset splited in train and test.")

    # Creating and training model
    RF_model = housepricing_pipeline.fit_random_forest(X_train=X_train, y_train=y_train)
    RF_model.fit(X_train, y_train)
    logger.info("Random Forest model was fitted.")

    # Model making a prediction on test data, and persisting the model
    # predictor = ModelPredictor(model=RF_model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, trained_model_dir=TRAINED_MODEL_DIR, file_save_name=PIPELINE_SAVE_FILE)
    y_pred = housepricing_pipeline.predict(X_test=X_test)
    evaluation_metrics = housepricing_pipeline.get_evaluation_metrics(y_test=y_test)
    logger.info("Model Evaluation Metrics: ")
    logger.debug(evaluation_metrics)
    housepricing_pipeline.persist_model(trained_model_dir=TRAINED_MODEL_DIR, file_save_name=PIPELINE_SAVE_FILE)
    logger.info("Model generated into .pkl file.")

    # Predictions with new information
    logger.info("Making predictions with new information")
    new_data_pred = pd.DataFrame([[0.06905, 0.0, 2.18, 0, 0.458, 7.147, 54.2, 6.0622, 3, 222.0, 18.7, 396.90, 5.33]])
    housepricing_pipeline.make_prediction(X_values=new_data_pred, features=FEATURES, selected_features=SELECTED_FEATURES)
    logger.info("Prediction example 1. Done.")
    new_data_pred = pd.DataFrame([[0.02729, 0.0, 7.07, 0, 0.469, 7.185, 61.1, 4.9671, 2, 242.0, 17.8, 392.83, 4.03]])
    housepricing_pipeline.make_prediction(X_values=new_data_pred, features=FEATURES, selected_features=SELECTED_FEATURES)
    logger.info("Prediction example 2. Done.")
    new_data = X_test.tail(10)
    housepricing_pipeline.make_prediction(X_values=new_data, selected_features=SELECTED_FEATURES)
    logger.info("Prediction example 3. Done.")
    # print(list(y_test.tail(10)))
