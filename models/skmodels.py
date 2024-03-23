from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from models.basemodel import BaseModel,Scorers
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

class model():
    def __init__(self,kfold=5):
        self.k = kfold
        self.modelnames = ["LinearRegression","Ridge", "Lasso","RandomForest","XGBoost","GradientBoosting","AdaBoost","DecisionTree"]


    def name2model(self,name):
        if name == "LinearRegression":
            return LinearRegression()
        if name == "Ridge":
            return Ridge()
        if name == "Lasso":
            return Lasso()
        if name == "DecisionTree":
            return DecisionTreeClassifier()
        if name == "AdaBoost":
            return AdaBoostClassifier()
        if name == "GradientBoosting":
            return GradientBoostingClassifier()
        if name == "RandomForest":
            return RandomForestClassifier()
        if name == "XGBoost":
            return XGBClassifier()
        
    
    def cross_validate(self,modelname,X,y):

        scores = []
        aucs = []
        records = []
        kf = KFold(n_splits=self.k, shuffle=True, random_state=42)
        for i,(train_index, test_index) in enumerate(kf.split(X,y)):
            scorer = Scorers()
            # Get the training and test data for this fold
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Train and evaluate your model on this fold
            model = self.name2model(modelname)
            
            model.fit(X_train, y_train)
                
            score = model.score(X_test, y_test)
            scores.append(score)
            print("Model score on test data (fold {0:d}): {1:.4f}".format(i, score))
            
            y_prob = model.predict(X_test)
            print(f"sum y_prob obtain : {sum(y_prob)}")
            #auc = scorer.score(y_test, y_prob)
            #aucs.append(auc)
            #print("Model AUC on test data (fold {0:d}): {1:.4f}".format(i, auc))

        scores = np.array(scores)
        #aucs = np.array(aucs)
        print("Average ACC: {:.4f} pm {:.4f}".format(scores.mean(), scores.std()))
        #print("Average AUC: {0:.4f}, {1:.4f}".format(aucs.mean(), aucs.std()))
        return records
