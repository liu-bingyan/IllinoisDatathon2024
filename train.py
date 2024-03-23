import pandas as pd
from preprocessor import preprocessor
from models.skmodels import model
from load_dataframes import load_dataframes

df = load_dataframes(r'C:\Users\bingy\Box\Data Set for Competition')

# preprocess
pc = preprocessor()
df = pc.process(df)

# train
y = df['reason']
X = df.drop(columns = ['reason'])

models = model()
modelnames = [#"LinearRegression",
              #"Ridge", 
              #"Lasso",
              "RandomForest",
              "XGBoost",
              "GradientBoosting",
              "AdaBoost",
              "DecisionTree"
              ]

for name in modelnames: 
    print(f"Training model : {name}")
    models.cross_validate(name,X, y)
