import pandas as pd
from utils.preprocessor import preprocessor
from models.skmodels_regressor import model
from utils.load_dataframes import load_dataframes

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore", category=UserWarning) 



def load_data(mode = 'recall', debug= True):
    path = "data/data_preprocessed_"+mode+".csv"
    df = pd.read_csv(path)

    X = df.drop(columns = ['recall'])
    y = df['recall']
    print(f"Data loaded, X.shape : {X.shape}, y.shape : {y.shape}")
    return X,y

def main(mode='recall'):
    X,y = load_data(mode=mode)
    print(X.sample(5))
    
    modelnames = ["LinearRegression",
                #"Ridge", 
                #"Lasso",
                #"RandomForest",
                #"XGBoost",
                #"GradientBoosting",
                #"AdaBoost",
                #"DecisionTree"
                ]

    models = model()
    for name in modelnames: 
        print(f"Training model : {name}")
        models.cross_validate(name, mode, X, y)


if __name__ == "__main__":
    for mode in ['recall']:
        main(mode=mode)

    