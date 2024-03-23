import pandas as pd
from utils.preprocessor import preprocessor
from models.skmodels import model
from utils.load_dataframes import load_dataframes

def write_dataframes():
    df = load_dataframes(r'C:\Users\bingy\Box\Data Set for Competition')
    df.to_csv("data/data.csv", index = False)

def main():
    df = load_dataframes("data/data.csv")

    # preprocess
    pc = preprocessor()
    df = pc.process(df)

    # train
    y = df['reason']
    X = df.drop(columns = ['reason'])


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
    #for name in models.skmodels.ModelNames:
    for name in modelnames: 
        print(f"Training model : {name}")
        models.cross_validate(name, X, y)

def test():
    pass
    

if __name__ == "__main__":
    #main()
    #test()
    write_dataframes()