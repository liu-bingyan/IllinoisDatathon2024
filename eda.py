import glob
import pandas as pd

folder_path = r'C:\Users\bingy\Box\Data Set for Competition'
def load_dataframes(self, folder_path):
        csv_files = glob.glob(folder_path + '/*.csv')
        dataframes = []
        for file in csv_files:
            df = pd.read_csv(file)
            dataframes.append(df)
        return dataframes

class EDA():
    def __init__(self, dataframes):
        self.dataframes = dataframes    
    ### some simple EDA functions
    def print_types(self):
        for df in self.dataframes:
            print(df.dtypes)
    def print_shape(self):
        for df in self.dataframes:
            print(df.shape)

eda = EDA(load_dataframes(folder_path))