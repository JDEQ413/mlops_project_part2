import os
import sys

# api = KaggleApi()
# api.authenticate()

# api.dataset_download_file('fedesoriano/the-boston-houseprice-data','boston.csv')
# if os.path.isfile('boston.csv'):            # Searches for the new file downloaded inside default data directory
#    #os.remove('boston.csv')                 # ,and deletes it
#    print("Shi tah!")
# else:
#    print("Nanchis")

# import opendatasets as od
# import pandas as pd

# od.download('https://www.kaggle.com/datasets/fedesoriano/the-boston-houseprice-data', force=True)

# print(os.get_exec_path())
# print(os.path.dirname(os.path.realpath(__file__)))
# os.chdir('../bci_framework')

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
print(parent_dir)
print(os.path.abspath(os.path.join(parent_dir, "..")))
sys.path.append(current_dir)
relative_path = "mlops_project\\models"
model_path = os.path.join(parent_dir, relative_path)
print(model_path)
