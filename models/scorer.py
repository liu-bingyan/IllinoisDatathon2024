from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, log_loss, roc_auc_score

import numpy as np


class Scorers():
    def __init__(self):
        self.accuracys = []
        self.f1s = []
        self.log_losses = []
        self.roc_auc = []
        pass
    
    def eval(self,y_true, y_pred,verbose = True):
        self.accuracys.append(accuracy_score(y_true, y_pred))
        self.f1s.append(f1_score(y_true, y_pred))
        self.log_losses.append(log_loss(y_true, y_pred))
        self.roc_auc.append(roc_auc_score(y_true, y_pred))
        
        if verbose:
            print(f"accuracy : {self.accuracys[-1]}")
            print(f"f1 : {self.f1s[-1]}")
            print(f"log_loss : {self.log_losses[-1]}")
            print(f"roc_auc : {self.roc_auc[-1]}")
        pass
    
    def print_scores(self):
        print(f"accuracy : {np.array(self.accuracys).mean(), np.array(self.accuracys).std()}")
        print(f"f1 : {np.array(self.f1s).mean(), np.array(self.f1s).std()}")
        print(f"log_loss : {np.array(self.log_losses).mean(), np.array(self.log_losses).std()}")
        print(f"roc_auc : {np.array(self.roc_auc).mean(), np.array(self.roc_auc).std()}")
        pass