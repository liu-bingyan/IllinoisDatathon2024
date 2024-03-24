import pandas as pd
from utils.preprocessor import preprocessor
from models.skmodels import model
from utils.load_dataframes import load_dataframes

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore", category=UserWarning) 


def write_dataframes():
    df = load_dataframes(r'C:\Users\bingy\Box\Data Set for Competition') # change this to your own path
    df.to_csv("data/data.csv", index = False)

def load_data(preprocess = False, mode = 'drop'):
    if not preprocess:
        path = "data/data.csv" 
    else:
        path = "data/data_preprocessed_"+mode+".csv"

    df = pd.read_csv(path)

    if not preprocess:
        pc = preprocessor(mode=mode)
        df = pc.process(df)
        pc.save_files(df)

    y = df['reason']
    X = df.drop(columns = ['reason'])
    print(f"Data loaded, X.shape : {X.shape}, y.shape : {y.shape}")
    return X,y

def main(preprocess = False,mode='drop'):
    X,y = load_data(preprocess = preprocess,mode=mode)
    
    modelnames = [#"LinearRegression",
                #"Ridge", 
                #"Lasso",
                #"RandomForest",
                "XGBoost",
                #"GradientBoosting",
                #"AdaBoost",
                #"DecisionTree"
                ]

    models = model()
    for name in modelnames: 
        print(f"Training model : {name}")
        models.cross_validate(name, X, y)
    

if __name__ == "__main__":
    main(preprocess = True,mode = 'drop')

    