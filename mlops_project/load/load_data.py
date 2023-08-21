import os
import shutil
from pathlib import Path

import opendatasets as od
import pandas as pd


class DataRetriever():
    """
    A class to retrieve data from Kaggle specifically.

    Parameters:
        paths_list (list of str): Requires a list of strings with paths to download and store files addecuately.
            [0] = MAIN_DIR
            [1] = DATASETS_DIR
            [2] = KAGGLE_URL
            [3] = KAGGLE_LOCAL_DIR
            [4] = DATA_RETRIEVED

    Attributes:
        MAIN_DIR (str): 1st element of 'paths_list'. Main (root) directory of the project.
        DATASETS_DIR (str): 1st element of 'paths_list'. Directory where data retrieved is going to be stored.
        KAGGLE_URL (str): 2nd element of 'paths_list'. URL from which dataset is going to be downloaded.
        KAGGLE_LOCAL_DIR (str): 3rd element of 'paths_list'. Folder extracted from KAGGLE_URL, where dataset is dowonloaded by default.
        DATA_RETRIEVED: 4th element of 'paths_list'. Name for the final retrieved dataset.

    """

    def __init__(self, paths_list: list) -> None:
        self.DATASETS_DIR = paths_list[0]
        self.KAGGLE_URL = paths_list[1]
        self.KAGGLE_LOCAL_DIR = paths_list[2]
        self.DATA_RETRIEVED = paths_list[3]

    def retrieve_data(self) -> bool:
        # Downloads dataset from kaggle with pre-defined structure (folder)

        od.download(self.KAGGLE_URL, force=True)

        # Finds the recently downloaded file
        if not os.path.isdir("/" + self.KAGGLE_LOCAL_DIR + "/"):
            paths = sorted(Path(self.KAGGLE_LOCAL_DIR + "/").iterdir(), key=os.path.getmtime)
        else:
            print("Directory could not be found: " + self.KAGGLE_LOCAL_DIR)
            return False
        path_new_file = str(paths[-1])
        name_new_file = str(path_new_file).split('\\')[-1]
        path_new_file = "./" + str(path_new_file).split('\\')[0] + "/" + str(path_new_file).split('\\')[-1]
        print("Dataset downloaded: " + path_new_file)

        # Moves the file to default data directory instead of downloaded folder
        if os.path.isfile(self.DATASETS_DIR + name_new_file):            # Searches for the new file downloaded inside default data directory
            os.remove(self.DATASETS_DIR + name_new_file)                 # ,and deletes it
        if os.path.isfile(self.DATASETS_DIR + self.DATA_RETRIEVED):           # Searches for any old file with FILE_NAME specified
            os.remove(self.DATASETS_DIR + self.DATA_RETRIEVED)                # ,and deletes it too
        os.rename(path_new_file, self.DATASETS_DIR + self.DATA_RETRIEVED)     # Finally, moves downloaded file to default datasets folder
        print("And stored in: " + self.DATASETS_DIR + self.DATA_RETRIEVED)
        shutil.rmtree(self.KAGGLE_LOCAL_DIR)

        return True

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.DATASETS_DIR + self.DATA_RETRIEVED, delimiter=",")
        return df

# Usage example:
