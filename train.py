import pandas as pd
from utils.preprocessor import preprocessor
from models.skmodels import model
from utils.load_dataframes import load_dataframes

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore", category=UserWarning) 


def write_dataframes():
    df = load_dataframes(r'C:\Users\bingy\Box\Data Set for Competition') # change this to your own path
    df.to_csv("data/data.csv", index = False)

def load_data(mode = 'drop', debug= True):
    path = "data/data_preprocessed_"+mode+".csv"
    df = pd.read_csv(path)
    y = df['reason']
    X = df.drop(columns = ['reason','resolved','no_of_accounts_with_syf_18_march',
                           'account_balance_18_march','account_status_18_march',
                           'card_activation_status_18_march','eservice_ind_18_march',
                           'ebill_enrolled_status_18_march','auto_pay_enrolled_status_18_march'
                           ])
    print(f"Data loaded, X.shape : {X.shape}, y.shape : {y.shape}")
    return X,y

def main(mode='drop'):
    X,y = load_data(mode=mode)
    print(X.sample(5))
    
    modelnames = [#"LinearRegression",
                #"Ridge", 
                #"Lasso",
                "RandomForest",
                "XGBoost",
                #"GradientBoosting",
                #"AdaBoost",
                "DecisionTree"
                ]

    models = model()
    for name in modelnames: 
        print(f"Training model : {name}")
        models.cross_validate(name, mode, X, y)


if __name__ == "__main__":
    for mode in ['drop']:#,'tail']:#['drop','length','TR', 'multihot']:
        main(mode=mode)

    