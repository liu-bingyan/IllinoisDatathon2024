# load data
import glob
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

modelnames = ["LinearRegression","RandomForest","XGBoost"]
for name in modelnames:
    if name == "LinearRegression":
        newmodel = model(name)
        newmodel.cross_validate(X,y)
