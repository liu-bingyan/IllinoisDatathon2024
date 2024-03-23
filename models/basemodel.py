from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, log_loss, roc_auc_score
import numpy as np


class BaseModel():
    def __init__(self):
        pass
    def fit(self, X, y):
        pass
    def score(self, X, y): #accuracy and auc 
        pass
    def cross_validate(self, X, y):
        pass

class Scorers():
    def __init__(self):
        pass
    def score(self,y_true, y_pred):
        return roc_auc_score(y_true, y_pred)