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
    
    #def show_results(self):
        # check columns equality
        # ...
        # ...

    ### some simple EDA functions
    def resolved_to_binary(self):
        for df in self.dataframes:
            df['resolved_binary'] = df['resolved'].apply(lambda x: 1 if x == 'resolved' else 0)
        return df

    def check_columns_equality(self, column1, column2):
        for df in self.dataframes:
            new_column_name = column1 + "_compare"
            df[new_column_name] = df.apply(lambda x: 1 if x[column1] == x[column2] else 0, axis=1)
        return df

    def print_types(self):
        for df in self.dataframes:
            print(df.dtypes)
    def print_shape(self):
        for df in self.dataframes:
            print(df.shape)

if __name__ == "__main__":
    eda = EDA(load_dataframes(folder_path))