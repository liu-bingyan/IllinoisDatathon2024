import pandas as pd
from utils.preprocessor import preprocessor
from models.skmodels import model
from utils.load_dataframes import load_dataframes

def write_dataframes():
    df = load_dataframes(r'C:\Users\bingy\Box\Data Set for Competition') # change this to your own path
    df.to_csv("data/data.csv", index = False)

def load_data(sample = 0):
    if sample==0:
        df = pd.read_csv("data/data.csv")
    else:
        df = pd.read_csv("data/data.csv", nrows = sample)

    pc = preprocessor()
    df = pc.process(df)
    y = df['reason']
    X = df.drop(columns = ['reason'])
    print(f"Data loaded, X.shape : {X.shape}, y.shape : {y.shape}")
    return X,y

def main():
    X,y = load_data(2000)

    modelnames = [#"LinearRegression",
                #"Ridge", 
                #"Lasso",
                #"RandomForest",
                #"XGBoost",
                #"GradientBoosting",
                #"AdaBoost",
                "DecisionTree"
                ]

    models = model()
    for name in modelnames: 
        print(f"Training model : {name}")
        models.cross_validate(name, X, y)
    

if __name__ == "__main__":
    main()