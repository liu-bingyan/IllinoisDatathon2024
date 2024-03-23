from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

modelnames = ["LinearRegression","RandomForest","XGBoost"]

class model():
    def __init__(self,name,kfold=5):
        self.modelname = name
        self.k = kfold

    def name2model(self,name):
        if name == "LinearRegression":
            return LinearRegression()
        if name == "RandomForest":
            return RandomForestRegressor()
        if name == "XGBoost":
            return XGBRegressor()
        
    
    def cross_validate(self,X,y):
        scores = []
        rmses = []

        kf = KFold(n_splits=self.k, shuffle=True, random_state=42)

        # Iterate over the folds
        for i,(train_index, test_index) in enumerate(kf.split(X,y)):
            # Get the training and test data for this fold
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Train and evaluate your model on this fold
            model = self.name2model(self.modelname)
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            scores.append(score)
            print("Model score on test data (fold {0:d}): {1:.4f}".format(i, score))
            
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred, squared=False))
            rmses.append(rmse)
            print("RMSE on test data (fold {0:d}): {1:.4f}".format(i, rmse))

        scores = np.array(scores)
        rmses = np.array(rmses)
        print("Average validation score: {:.4f} pm {:.4f}".format(scores.mean(), scores.std()))
        print("Average RMSE: {0:.4f}, {1:.4f}".format(rmses.mean(), rmses.std()))

if __name__ == "__main__":
    ourmodel = model()
    ourmodel.cross_validate(X,y)