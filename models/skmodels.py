from sklearn.model_selection import StratifiedKFold
from models.scorer import Scorers
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

ModelNames = ["LinearRegression","Ridge", "Lasso","RandomForest","XGBoost","GradientBoosting","AdaBoost","DecisionTree"] 

def name2model(name):
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
    raise ValueError("Model name not found")

class model():
    def __init__(self,kfold=5):
        self.k = kfold
        pass
    
    def cross_validate(self,modelname,X,y):

        scorer = Scorers()
        
        kf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=42)
        for i,(train_index, test_index) in enumerate(kf.split(X,y)):
 
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            model = name2model(modelname)
            
            model.fit(X_train, y_train)

            y_prob = model.predict(X_test)

            scorer.eval(y_test, y_prob,verbose = True)           

        scores = np.array(scores)
        print("Average ACC: {:.4f} pm {:.4f}".format(scores.mean(), scores.std()))
        
        scorer.print_results()
        return 