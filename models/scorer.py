from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, log_loss, roc_auc_score

import numpy as np
from datetime import datetime
import pandas as pd


class Scorers():
    def __init__(self):
        self.accuracys = []
        self.f1s = []
        self.log_losses = []
        self.roc_auc = []
        pass
    
    def eval(self,y_true, y_pred,y_prob,verbose = True):
        logloss = log_loss(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob, multi_class='ovo', average="macro")
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted")  # use here macro or weighted?

        self.accuracys.append(acc)
        self.f1s.append(f1)
        self.log_losses.append(logloss)
        self.roc_auc.append(auc)
        
        if verbose:
            self.print()
        pass
    
    def print(self):
        print(f"accuracy : {self.accuracys[-1]}", f"f1 : {self.f1s[-1]}", f"log_loss : {self.log_losses[-1]}", f"roc_auc : {self.roc_auc[-1]}")
        pass

    def print_scores(self):
        print(f"accuracy : {np.array(self.accuracys).mean(), np.array(self.accuracys).std()}")
        print(f"f1 : {np.array(self.f1s).mean(), np.array(self.f1s).std()}")
        print(f"log_loss : {np.array(self.log_losses).mean(), np.array(self.log_losses).std()}")
        print(f"roc_auc : {np.array(self.roc_auc).mean(), np.array(self.roc_auc).std()}")
        pass
    def save_results(self,name):
        df =pd.DataFrame(columns = ["accuracy","f1","log_loss","roc_auc"])
        df["accuracy"] = self.accuracys
        df["f1"] = self.f1s
        df["log_loss"] = self.log_losses
        df["roc_auc"] = self.roc_auc
        df.loc["mean"] = df.mean()
        df.index = df.index.fillna("mean")
        df.loc["std"] = df.std()
        df.index = df.index.fillna("std")
        print(df)
        timestamp = str(datetime.now()).replace(":", "_")  # Replace colon with underscore in the timestamp
        df.to_csv(f"results/{name}_{timestamp}.csv")
        pass



if __name__ == "__main__":
    scorer = Scorers()
    y_true = np.array([2, 0, 1, 2])
    y_pred_proba = np.array([[0.2, 0.3, 0.5], [0.7, 0.1, 0.2], [0.3, 0.4, 0.3], [0.1, 0.2, 0.7]])
    y_pred = np.argmax(y_pred_proba, axis=1)
    scorer.eval(y_true, y_pred,y_pred_proba)
    scorer.save_results("test")
