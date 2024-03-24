import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
import pickle


class preprocessor():
    def __init__(self,mode = 'drop',debug = False,onehot=False):
        self.debug = debug
        self.onehot = onehot
        self.mode = mode

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
        if self.debug:
            print("start processing data") 
            print(f"    original shape : {df.shape}")

        df = self.process_na(df)
        if self.debug:
            print(f"    shape after processing na : {df.shape}")

        df = self.classify_columns(df)
        if self.debug:
            print(f"    shape after classifying columns : {df.shape}")

        df = self.process_reasons(df)
        if self.debug:
            print(f"    shape after processing reasons : {df.shape}")

        df = self.process_mos(df, mode = "drop")
        if self.debug:
            print(f"    shape after processing mos : {df.shape}")

        df = self.encode_labels(df)
        if self.debug:
            print(f"    shape after encod_labels : {df.shape}")

        print('Data Preprocessing Done')
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
        if self.debug:
            print(f"        added time and string columns : {self.time_columns}")

        # binary columns
        df["resolved"] = df["resolved"].apply(lambda x: 1 if x == "resolved" else 0)
        self.binary_columns = ['resolved',
                               "eservice_ind_13_march","eservice_ind_18_march",
                               "auto_pay_enrolled_status_13_march","auto_pay_enrolled_status_18_march"]
        if self.debug:
            print(f"        added binary columns : {self.binary_columns}")

        # interger columns
        df["delinquency_current_13_march"] = df["delinquency_history_13_march"].apply(lambda x: int(x[1]))
        df["delinquency_prior_13_march"] = df["delinquency_history_13_march"].apply(lambda x: int(x[2]))
        df = df.drop(columns = ["delinquency_history_13_march"])
        df["delinquency_current_18_march"] = df["delinquency_history_18_march"].apply(lambda x: int(x[1]))
        df["delinquency_prior_18_march"] = df["delinquency_history_18_march"].apply(lambda x: int(x[2]))
        df = df.drop(columns = ["delinquency_history_18_march"])

        df = df[df["account_open_date_13_march"]==df["account_open_date_18_march"]]
        s = df["account_open_date_13_march"].apply(lambda x: pd.to_datetime(x, format = "%m/%d/%Y"))
        df["account_history_length"] = (df["call_day"] - s).apply(lambda x : x.days)
        df = df.drop(columns = ["account_open_date_18_march"])
        df = df.drop(columns = ["account_open_date_13_march"])
        self.interger_columns = ["delinquency_current_13_march","delinquency_current_18_march",
                                 "delinquency_prior_13_march","delinquency_prior_18_march",
                                 "no_of_accounts_with_syf_13_march","no_of_accounts_with_syf_18_march",
                                 "account_history_length",]
        if self.debug:
            print(f"        added interger columns : {self.interger_columns}")
        

        self.numerical_columns = ["account_balance_13_march","account_balance_18_march"]
        self.categorical_columns = ["retailer_code",
                                    "reason",
                                    "mos",
                                    "ebill_enrolled_status_13_march","ebill_enrolled_status_18_march",
                                    "account_status_13_march","account_status_18_march",
                                    "card_activation_status_13_march","card_activation_status_18_march",]
        if self.debug:
            print(f"        added numerical columns : {self.numerical_columns}")
            print(f"        added categorical columns : {self.categorical_columns}")
        

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

    def process_reasons(self,df):
        # remove most comment reasons and rare reasons

        vc = df['reason'].value_counts()
        #common_reasons = vc
        rare_reasons = vc[vc<5].index
        df.drop(df[df['reason'].isin(rare_reasons)].index, inplace = True)
        if self.debug: 
            print(f"    removed rare reasons, details : {rare_reasons[:5]}...")
        return df
   
    def process_mos(self,df):
        # correct false resolved
        ends_without_TR = (lambda x: "TR"!=x.split(" ")[-1])
        s = df['mos'].apply(ends_without_TR)
        df['resolved'] = df['resolved'] & s

        # process the mos 
        mode = self.mode
        self.mos_encoder = None
        if mode == "drop":
            df = df.drop(columns = ["mos"])
            self.categorical_columns.remove("mos")
        elif mode == "length":
            df["mos"] = df["mos"].apply(lambda x: len(x))
            self.interger_columns.append("mos_length")
        elif mode == "TR":
            df["mos"] = df["mos"].apply(lambda x: 1 if "TR" == x.split(" ")[-1] else 0)
            self.binary_columns.append("mos_TR")

        elif mode == "tail":
            df["mos"] = df["mos"].apply(lambda x: int(x[-1]))
            self.categorical_columns.append("mos_tail")
        elif mode == "multihot":
            df["mos"] = df["mos"].apply(lambda x: x.split(" "))
            mlb = MultiLabelBinarizer()
            encoded_df = pd.DataFrame(mlb.fit_transform(df["mos"]), columns=mlb.classes_)
            df = pd.concat([df.drop("mos",axis=1),encoded_df],axis = 1)

            self.binary_columns += list(mlb.classes_)
            self.mos_encoder = mlb
        return df
    
    def encode_labels(self,df):
        #categorical columns : ['retailer_code', 'reason', 'ebill_enrolled_status_13_march', 'ebill_enrolled_status_18_march', 'account_status_13_march', 'account_status_18_march', 'card_activation_status_13_march', 'card_activation_status_18_march']
        self.reason_encoder = LabelEncoder()
        if self.mos_mode == "tail":
            self.reason_encoder.fit(pd.concat([df["reason"],df["mos_tail"]]))
            df['reason'] = self.reason_encoder.transform(df["reason"])
            df['mos_tail'] = self.reason_encoder.transform(df["mos_tail"])
        else:
            df['reason'] = self.reason_encoder.fit_transform(df["reason"])
            

        self.retailer_code_encoder = LabelEncoder()
        df["retailer_code"] = self.retailer_code_encoder.fit_transform(df["retailer_code"])      

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

        # string columns
        df.drop(columns = ["serial","call_key"], inplace = True)
        self.string_columns.remove("serial")
        self.string_columns.remove("call_key")

        # time columns
        df.drop(columns = ["call_day","call_time"], inplace = True)
        self.time_columns.remove("call_day")
        self.time_columns.remove("call_time")

        self.save_encorders()

        if self.onehot:
            df = pd.get_dummies(df, columns = self.categorical_columns)
            self.categorical_columns = []
        return df 
    
    def save_encorders(self):
        # Save encoders to a file
        encoders = {
            "mos_mode": self.mos_mode, # "drop","length","TR","tail","multihot"
            "mos_encoder": self.mos_encoder,
            "reason_encoder": self.reason_encoder,
            "retailer_code_encoder": self.retailer_code_encoder,
            "account_status_encoder": self.account_status_encoder,
            "ebill_enrolled_status_encoder": self.ebill_enrolled_status_encoder,
            "card_activation_status_encoder": self.card_activation_status_encoder
        }

        with open("data/"+self.mode+"_encoders.pkl", "wb") as file:
            pickle.dump(encoders, file)

    def save_files(self,df):
        df.to_csv("data/data_preprocessed_"+self.mode+".csv", index = False)
        return df
    
    
    
if __name__ == "__main__":
    TESTCASE = "A"
    if TESTCASE == "A":
        df = pd.read_csv("data/data.csv",nrows=10)
        pp = preprocessor(debug=True,mode="TR")
        df = pp.process(df)
     
        pp.save_file(df)

    if TESTCASE == "B":
        with open("data/encoders_TR.pkl", "rb") as file:
            encoders = pickle.load(file)
            for k,v in encoders.items():
                print(k,v)
                if k == "ebill_enrolled_status_encoder":
                    print(v.classes_)
