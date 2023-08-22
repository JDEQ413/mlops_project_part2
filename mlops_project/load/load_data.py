import os

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from utilities.custom_loging import CustomLogging

logger = CustomLogging()
logger = logger.Create_Logger(logger_name='load_data.log')


class DataRetriever():
    """
    A class to retrieve data from Kaggle specifically.

    Parameters:
        paths_list (list of str): Requires a list of strings with paths to download and store files addecuately.
            [0] = MAIN_DIR
            [1] = DATASETS_DIR
            [2] = KAGGLE_URL
            [3] = KAGGLE_FILE
            [4] = DATA_RETRIEVED

    Attributes:
        MAIN_DIR (str): 1st element of 'paths_list'. Main (root) directory of the project.
        DATASETS_DIR (str): 2nd element of 'paths_list'. Directory where data retrieved is going to be stored.
        KAGGLE_URL (str): 3rd element of 'paths_list'. URL from which dataset is going to be downloaded.
        KAGGLE_FILE (str): 4th element of 'paths_list'. Name of the file to retrieve from KAGGLE_URL.
        DATA_RETRIEVED: 5th element of 'paths_list'. Name for the final retrieved dataset.

    """

    def __init__(self, paths_list: list) -> None:
        self.MAIN_DIR = paths_list[0]
        self.DATASETS_DIR = paths_list[1]
        self.KAGGLE_URL = paths_list[2]
        self.KAGGLE_FILE = paths_list[3]
        self.DATA_RETRIEVED = paths_list[4]

        self.api = KaggleApi()
        self.api.authenticate()
        logger.info("Kaggle authentication succesfully made.")

    def retrieve_data(self) -> bool:
        # Downloads dataset from kaggle with pre-defined structure (folder)
        # od.download(self.KAGGLE_URL, force=True)
        self.api.dataset_download_file(self.KAGGLE_URL, self.KAGGLE_FILE)

        # Moves the file to default data directory instead of downloaded folder
        if os.path.isfile(self.DATASETS_DIR + self.KAGGLE_FILE):            # Searches for the new file downloaded inside default data directory
            os.remove(self.DATASETS_DIR + self.KAGGLE_FILE)                 # ,and deletes it
            logger.warning("Previous data.csv existing file was deleted.")
        if os.path.isfile(self.DATASETS_DIR + self.DATA_RETRIEVED):           # Searches for any old file with FILE_NAME specified
            os.remove(self.DATASETS_DIR + self.DATA_RETRIEVED)                # ,and deletes it too
        os.rename(self.KAGGLE_FILE, self.DATASETS_DIR + self.DATA_RETRIEVED)     # Finally, moves downloaded file to default datasets folder
        logger.info("Dataset retrieved and stored in: " + self.DATASETS_DIR + self.DATA_RETRIEVED)

        return True

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.DATASETS_DIR + self.DATA_RETRIEVED, delimiter=",")
        logger.info("Dataset was loaded successfully.")
        return df

# Usage example:
