import pandas as pd
import glob

def load_dataframes(folder_path):
        csv_files = glob.glob(folder_path + '/*.csv')
        dataframes = []
        for file in csv_files:
            df = pd.read_csv(file)
            dataframes.append(df)
        df = pd.concat(dataframes)
        return df

if __name__ == "__main__":
    folder_path = r'C:\Users\bingy\Box\Data Set for Competition'
    df = load_dataframes(folder_path)
    print(df.shape)
