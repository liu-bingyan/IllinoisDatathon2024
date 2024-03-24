import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, 
from sklearn.preprocessing import MultiLabelBinarizer


class preprocessor():
    def __init__(self,debug = False):
        self.debug = debug
        self.binary_columns = []
        self.interger_columns = []
        self.numerical_columns = []

        self.time_columns = []
        self.categorical_columns = []
        self.string_columns = []

    def print_columns_classes(self):
        print(f"binary columns : {self.binary_columns}")
        print(f"interger columns : {self.interger_columns}")
        print(f"numerical columns : {self.numerical_columns}")
        print(f"categorical columns : {self.categorical_columns}")
        print(f"string columns : {self.string_columns}")
        print(f"time columns : {self.time_columns}")
        return   

    def process(self,df):
        df = self.process_na(df)
        df = self.classify_columns(df)
        df = self.main(df)
        return df

    def classify_columns(self,df):
        # string and time columns
        df["timestamp"] = df["timestamp_call_key"].apply(lambda x: x[:12])
        df["call_day"] = df["timestamp"].apply(lambda x: x[:6]).apply(lambda x: pd.to_datetime(x, format = "%y%m%d"))
        df["call_time"] = df['timestamp'].apply(lambda x: x[6:]).apply(lambda x: pd.to_datetime(x, format = "%H%M%S"))
        df["call_key"] = df["timestamp_call_key"].apply(lambda x: x[-8:])
        df = df.drop(columns = ["timestamp_call_key","timestamp"])
        self.time_columns = ["call_day", "call_time"]
        self.string_columns = ["serial", "call_key"]

        # binary columns
        df["resolved"] = df["resolved"].apply(lambda x: 1 if x == "resolved" else 0)
        self.binary_columns = ['resolved',
                               "eservice_ind_13_march","eservice_ind_18_march",
                               "auto_pay_enrolled_status_13_march","auto_pay_enrolled_status_18_march"]

        # interger columns
        df["delinquency_current_13_march"] = df["delinquency_history_13_march"].apply(lambda x: int(x[1]))
        df["delinquency_prior_13_march"] = df["delinquency_history_13_march"].apply(lambda x: int(x[2]))
        df = df.drop(columns = ["delinquency_history_13_march"])
        df["delinquency_current_18_march"] = df["delinquency_history_18_march"].apply(lambda x: int(x[1]))
        df["delinquency_prior_18_march"] = df["delinquency_history_18_march"].apply(lambda x: int(x[2]))
        df = df.drop(columns = ["delinquency_history_18_march"])
        df = df[df["account_open_date_13_march"]==df["account_open_date_18_march"]]
        s = df["account_open_date_13_march"].apply(lambda x: pd.to_datetime(x, format = "%m/%d/%Y"))
        df["account_history_length"] = (df["call_day"] - s).days
        df = df.drop(columns = ["account_open_date_18_march"])
        df = df.drop(columns = ["account_open_date_13_march"])
        self.interger_columns.append("account_history_length")
        self.interger_columns = ["delinquency_current_13_march","delinquency_current_18_march",
                                 "delinquency_prior_13_march","delinquency_prior_18_march",
                                 "no_of_accounts_with_syf_13_march","no_of_accounts_with_syf_18_march",
                               ]
        

        self.numerical_columns = ["account_balance_13_march","account_balance_18_march"]
        self.categorical_columns = ["retailer_code",
                                    "reason",
                                    "mos",
                                    "ebill_enrolled_status_13_march","ebill_enrolled_status_18_march",
                                    "account_status_13_march","account_status_18_march",
                                    "card_activation_status_13_march","card_activation_status_18_march",]

        

        return df
    
    def process_na(self,df):
        df['no_of_accounts_with_syf_13_march'] = df['no_of_accounts_with_syf_13_march'].fillna(0)
        df['no_of_accounts_with_syf_18_march'] = df['no_of_accounts_with_syf_18_march'].fillna(0)
        df['account_balance_13_march'] = df['account_balance_13_march'].fillna(0)
        df['account_balance_18_march'] = df['account_balance_18_march'].fillna(0)
        df['account_status_13_march'] = df['account_status_13_march'].fillna('N')
        df['account_status_18_march'] = df['account_status_18_march'].fillna('N')
        df['ebill_enrolled_status_13_march'] = df['ebill_enrolled_status_13_march'].fillna('N/A')
        df['ebill_enrolled_status_18_march'] = df['ebill_enrolled_status_18_march'].fillna('N/A')
        return df

    def main(self,df):   
        df = self.process_reasons(df)
        df = self.process_mos(df, mode = "drop")
        df = self.encode_labels(df)
        return df  
    
    def process_reasons(self,df):
        vc = df['reason'].value_counts()
        rare_reasons = vc[vc<5].index
        df.drop(df[df['reason'].isin(rare_reasons)].index, inplace = True)
        if self.debug: 
            print(f"removed rare reasons, details : {rare_reasons}")
        return df
   
    def process_mos(self,df, mode = "drop"):
        if mode == "drop":
            df = df.drop(columns = ["mos"])
        elif mode == "length":
            df["mos"] = df["mos"].apply(lambda x: len(x))
            self.interger_columns.append("mos_length")
        elif mode == "TR":
            df["mos"] = df["mos"].apply(lambda x: 1 if "TR" in x.split(" ") else 0)
            self.binary_columns.append("mos_TR")
        elif mode == "tail":
            df["mos"] = df["mos"].apply(lambda x: int(x[-1]))
            self.categorical_columns.append("mos_tail")
        elif mode == "onehot":
            df["mos"] = df["mos"].apply(lambda x: x.split(" "))
            mlb = MultiLabelBinarizer()
            encoded_df = pd.DataFrame(mlb.fit_transform(df["mos"]), columns=mlb.classes_)
            df = pd.concat([df.drop("mos",axis=1),encoded_df],axis = 1)
            self.binary_columns += list(mlb.classes_)
        return df
    
    def encode_labels(self,df):
        self.account_status_encoder = LabelEncoder()
        self.account_status_encoder.fit(pd.concat([df["account_status_13_march"],df["account_status_18_march"]]))
        for col in ["account_status_13_march","account_status_18_march"]:
            df[col] = self.account_status_encoder.transform(df[col])
        
        self.ebill_enrolled_status_encoder = LabelEncoder()
        self.ebill_enrolled_status_encoder.fit(pd.concat([df["ebill_enrolled_status_13_march"],df["ebill_enrolled_status_18_march"]]))
        for col in ["ebill_enrolled_status_13_march","ebill_enrolled_status_18_march"]:
            df[col] = self.ebill_enrolled_status_encoder.transform(df[col])

        self.card_activation_status_encoder = LabelEncoder()
        self.card_activation_status_encoder.fit(pd.concat([df["card_activation_status_13_march"],df["card_activation_status_18_march"]]))
        for col in ["card_activation_status_13_march","card_activation_status_18_march"]:
            df[col] = self.card_activation_status_encoder.transform(df[col])
 
        
if __name__ == "__main__":
    EDA = True
    if EDA:
        df = pd.read_csv("data/data.csv")
        df['account_open_date_13_march'] = pd.to_datetime(df['account_open_date_13_march'])
    else:

        df = pd.read_csv("data/data.csv")
        pp = preprocessor()
        df = pp.process(df)
        