# load data
import glob
import pandas as pd
from preprocessor import preprocessor
from models.skmodels import model
from load_dataframes import load_dataframes
from datetime import datetime

df3 = pd.read_csv("file_3_Mar18_Output_1.csv")
df4 = pd.read_csv("file_4_Mar18_Output_1.csv")
df = df3

# preprocess
pc = preprocessor()
print(datetime.now(), "starting preprocessing")
df = pc.process(df)
print(datetime.now(), "done preprocessing")

# train
print(datetime.now(), "about to select x and y")
y = df['reason']
X = df.drop(columns = ['reason'])
print(datetime.now(), "selected x and y")

# Comments on preliminary results (run on df3)

# LinearRegression              - runs okay, score about 0.6
# Ridge                         - appears to run okay, score about 0.6 but linalg warning ill-conditioned matrix
# Lasso                         - runs okay, score about 0.5
# ElasticNet                    - runs okay, score about 0.3

# AdaBoostClassifier            - (Jie gave up after about 5min)
# AdaBoostRegressor             - runs okay (about 60s to fit), score about 0.53

# DecisionTreeClassifier        - runs okay (about 30s to fit), score about 0.94
# DecisionTreeRegressor         - runs okay (about 30s to fit), score about 0.78

# GradientBoostingClassifier    - (Jie gave up after about 11min)
# GradientBoostingRegressor     - (Jie gave up after about 7min)

# RandomForestClassifier        - (Jie gave up after about 7min)
# RandomForestRegressor         - (Jie gave up after about 7min)

# XGBoostClassifier             - 
# XGBoostRegressor              - 

modelnames = ["LinearRegression", "Ridge", "Lasso", "ElasticNet", 
              "AdaBoostClassifier", "AdaBoostRegressor",
              "DecisionTreeClassifier", "DecisionTreeRegressor", 
              "GradientBoostingClassifier", "GradientBoostingRegressor", 
              "RandomForestClassifier", "RandomForestRegressor", 
              "XGBoostClassifier", "XGBoostRegressor"]

for name in modelnames:
    if name == "DecisionTreeClassifier":
        print(datetime.now(), "about to make new model =", name)
        newmodel = model(name)
        print(datetime.now(), "made new model")
        newmodel.cross_validate(X,y)
