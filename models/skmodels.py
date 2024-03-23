from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

import xgboost
from xgboost import XGBClassifier
from xgboost import XGBRegressor

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from models.basemodel import BaseModel,Scorers
import numpy as np
from datetime import datetime

modelnames = ["LinearRegression","RandomForest","XGBoost"]

class model():
    def __init__(self,name,kfold=5):
        self.modelname = name
        self.k = kfold

    def name2model(self,name):

        if name == "LinearRegression":
            return LinearRegression()
        if (name == "Ridge"):
            return Ridge()
        if (name == "Lasso"):
            return Lasso()
        if (name == "ElasticNet"):
            return ElasticNet()
        
        if (name == "AdaBoostClassifier"):
            return AdaBoostClassifier()
        if (name == "AdaBoostRegressor"):
            return AdaBoostRegressor()
        
        if (name == "DecisionTreeClassifier"):
            return DecisionTreeClassifier()
        if (name == "DecisionTreeRegressor"):
            return DecisionTreeRegressor()
        
        if (name == "GradientBoostingClassifier"):
            return GradientBoostingClassifier()
        if (name == "GradientBoostingRegressor"):
            return GradientBoostingRegressor()
        
        if name == "RandomForestClassifier":
            return RandomForestClassifier()
        if name == "RandomForestRegressor":
            return RandomForestRegressor()
        
        if name == "XGBoostClassifier":
            return XGBClassifier()
        if name == "XGBoostRegressor":
            return XGBRegressor()
        
        return None
        
    def cross_validate(self,X,y):
        print(datetime.now(), "entered cross validate")
        scores = []
        aucs = []
        print(datetime.now(), "about to make kfold train test split")
        kf = KFold(n_splits=self.k, shuffle=True, random_state=42)
        print(datetime.now(), "done making train test split")
        # Iterate over the folds
        for i,(train_index, test_index) in enumerate(kf.split(X,y)):
            scorer = Scorers()
            # Get the training and test data for this fold
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Train and evaluate your model on this fold
            model = self.name2model(self.modelname)
            
            print(datetime.now(), "about to fit")
            model.fit(X_train, y_train)

            text_representation = tree.export_text(model)
            print(text_representation)
            
            print(datetime.now(), "done fitting, about to score")
            score = model.score(X_test, y_test)
            
            print(datetime.now(), "done scoring")
            scores.append(score)
            
            print(datetime.now(), "Model score on test data (fold {0:d}): {1:.4f}".format(i, score))
            
            #y_prob = model.predict(X_test)
            #auc = scorer.score(y_test, y_prob)
            #aucs.append(auc)
            #print("Model AUC on test data (fold {0:d}): {1:.4f}".format(i, auc))

        scores = np.array(scores)
        #aucs = np.array(aucs)
        print("Average ACC: {:.4f} pm {:.4f}".format(scores.mean(), scores.std()))
        #print("Average AUC: {0:.4f}, {1:.4f}".format(aucs.mean(), aucs.std()))

    # def cross_validate_xgboost(self, X, y):
    #         print(datetime.now(), "entered cross validate xgboost")
    #         scores = []
    #         aucs = []
    #         print(datetime.now(), "about to make kfold train test split")
    #         kf = KFold(n_splits = self.k, shuffle = True, random_state = 42)
    #         print(datetime.now(), "done making train test split")
    #         # Iterate over the folds
    #         for i, (train_index, test_index) in enumerate(kf.split(X,y)):
    #             scorer = Scorers()
    #             # Get the training and test data for this fold
    #             X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    #             y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    #             # data = pandas.DataFrame(np.arange(12).reshape((4,3)), columns=['a', 'b', 'c'])
    #             # label = pandas.DataFrame(np.random.randint(2, size=4))
    #             dtrain = xgboost.DMatrix(X_train, label = y_train)
    #             dtest = xgboost.DMatrix(X_test, label = y_test)
                
    #             # Train and evaluate your model on this fold
    #             model = self.name2model(self.modelname)
    #             print(datetime.now(), "about to fit")
    #             # model.fit(X_train, y_train)
    #             param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
    #             param['nthread'] = 4
    #             param['eval_metric'] = 'auc'
    #             num_round = 10
    #             evallist = [(dtrain, 'train'), (dtest, 'eval')]
    #             bst = xgboost.train(param, dtrain, num_round, evallist)
    #             print(datetime.now(), "done fitting, about to score")
    #             # score = model.score(X_test, y_test)
    #             ypred = bst.predict(dtest)
    #             print(datetime.now(), "done scoring")
    #             scores.append(score)
    #             print(datetime.now(), "Model score on test data (fold {0:d}): {1:.4f}".format(i, score))
                
    #             #y_prob = model.predict(X_test)
    #             #auc = scorer.score(y_test, y_prob)
    #             #aucs.append(auc)
    #             #print("Model AUC on test data (fold {0:d}): {1:.4f}".format(i, auc))

    #         scores = np.array(scores)
    #         #aucs = np.array(aucs)
    #         print("Average ACC: {:.4f} pm {:.4f}".format(scores.mean(), scores.std()))
    #         #print("Average AUC: {0:.4f}, {1:.4f}".format(aucs.mean(), aucs.std()))
