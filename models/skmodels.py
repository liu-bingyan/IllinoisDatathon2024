from sklearn.model_selection import StratifiedKFold
from models.scorer import Scorers
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from datetime import datetime
import joblib
import pickle

ModelNames = ["LinearRegression","Ridge", "Lasso","RandomForest","XGBoost","GradientBoosting","AdaBoost","DecisionTree"] 

def name2model(name):
    if name == "LinearRegression":
        return LinearRegression()
    if name == "Ridge":
        return Ridge()
    if name == "Lasso":
        return Lasso()
    if name == "DecisionTree":
        return DecisionTreeClassifier(max_depth=5, min_samples_leaf=10)
    if name == "AdaBoost":
        return AdaBoostClassifier(n_estimators=100, random_state=42)
    if name == "GradientBoosting":
        return GradientBoostingClassifier(n_estimators=100, max_depth=5, min_samples_leaf=10)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=10)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=100, max_depth=5, min_child_weight=1)
    if name == "LightGBM":
        return LGBMClassifier(n_estimators=100, max_depth=5)
    raise ValueError("Model name not found")

class model():
    def __init__(self,kfold=5):
        self.k = kfold
        pass
    
    def cross_validate(self,modelname,mode,X,y):

        scorer = Scorers()
        
        kf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=42)
        for i,(train_index, test_index) in enumerate(kf.split(X,y)):
 
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            model = name2model(modelname)
            
            model.fit(X_train, y_train)
            #print(X_train.iloc[:5,:],y_train.iloc[:5])

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)

            self.save_model(model,path ="results/models",modelname=modelname,mode=mode)

            #print(f"y_porb_sum : {y_prob[:5,:].sum(axis=1)}")
            scorer.eval(y_test, y_pred, y_prob,verbose = True)           

        scorer.save_results(name = modelname,mode=mode)
        return 
    
    def save_model(self, model, path,modelname,mode):
        timestr = str(datetime.now()).replace(":", "_")
        path = path + "/" + modelname + "_" + mode + "_" + timestr+".pkl"
        with open(path, 'wb') as file:
            pickle.dump(model, file)