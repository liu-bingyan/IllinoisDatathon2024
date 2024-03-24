import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
import pickle


class preprocessor():
    def __init__(self,mode = 'drop',debug = True, onehot=False):
        self.debug = debug
        self.onehot = onehot
        self.mos_mode = mode

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

        df = self.correct_false_resolved(df)
        if self.debug:
            print(f"    shape after correcting false resolved : {df.shape}")

        df = self.process_reasons(df)
        if self.debug:
            print(f"    shape after processing reasons : {df.shape}")

        df = self.process_mos(df)
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

    def correct_false_resolved(self,df):
        # correct false resolved
        ends_without_TR = (lambda x: "TR"!=x.split(" ")[-1])
        s = df['mos'].apply(ends_without_TR)
        df['resolved'] = df['resolved'] & s
        return df

    def process_reasons(self,df):
        # remove most easy reasons and rare reasons
        def get_easy_reason(df):
            grouped = df.groupby('reason')
            result = grouped.agg({'resolved_truly': 'mean', 'resolved': 'mean'})
            vc = df['reason'].value_counts()
            dict = pd.read_csv('data/dictionary.csv')
            table = pd.merge(result, dict, left_index=True, right_on='Code')
            table = pd.merge(table, vc, left_on='Code', right_index=True)
            table.sort_values(by='resolved_truly', ascending=True)
            threshold = (table[table['Code']=='BA']['resolved_truly']).values[0]
            commen_table = table[table['resolved_truly'] >= threshold]
            commen_reasons = list(commen_table['Code'])
            return commen_reasons
        easy_reasons= ['AT', 'BA', 'CA', 'CB', 'ER', 'FR', 'Ls', 'PI', 'SP', 'VP', 'mm']
        df.drop(df[df['reason'].isin(easy_reasons)].index, inplace = True)
        if self.debug:
            print(f"    removed easy reasons, details : {easy_reasons[:5]}...")

        # remove rare reasons
        vc = df['reason'].value_counts()
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


        mode = self.mos_mode        # process the mos 
        self.mos_encoder = None
        if mode == "drop":
            df = df.drop(columns = ["mos"])
            self.categorical_columns.remove("mos")
        elif mode == "length":
            df["mos"] = df["mos"].apply(lambda x: len(x))
            self.interger_columns.append("mos_length")
            self.categorical_columns.remove("mos")
        elif mode == "TR":
            df["mos"] = df["mos"].apply(lambda x: 1 if "TR" == x.split(" ")[-1] else 0)
            self.binary_columns.append("mos_TR")
            self.categorical_columns.remove("mos")
        elif mode == "tail":
            df["mos"] = df["mos"].apply(lambda x: x.split(" ")[-1])
        elif mode == "head":
            df["mos"] = df["mos"].apply(lambda x: x.split(" ")[0])
        elif mode == "multihot":
            df["mos"] = df["mos"].apply(lambda x: x.split(" "))
            mlb = MultiLabelBinarizer()
            encoded_df = pd.DataFrame(mlb.fit_transform(df["mos"]), columns=mlb.classes_)
            df = pd.concat([df.drop("mos",axis=1).reset_index(),encoded_df],axis = 1)
            print(df.columns)

            self.binary_columns += list(mlb.classes_)
            self.categorical_columns.remove("mos")
            self.mos_encoder = mlb
        return df
    
    def encode_labels(self,df):
        #categorical columns : ['retailer_code', 'reason', 'ebill_enrolled_status_13_march', 'ebill_enrolled_status_18_march', 'account_status_13_march', 'account_status_18_march', 'card_activation_status_13_march', 'card_activation_status_18_march']
        self.reason_encoder = LabelEncoder()
        if (self.mos_mode == "tail") | (self.mos_mode == "head"):
            self.reason_encoder.fit(pd.concat([df["reason"],df["mos"]]))
            df['reason'] = self.reason_encoder.transform(df["reason"])
            df['mos'] = self.reason_encoder.transform(df["mos"])
        else:
            df['reason'] = self.reason_encoder.fit_transform(df["reason"])
        self.reason_encoder2 = LabelEncoder()
        df['reason'] = self.reason_encoder2.fit_transform(df["reason"])
            

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
            "reason_encoder2": self.reason_encoder2,
            "retailer_code_encoder": self.retailer_code_encoder,
            "account_status_encoder": self.account_status_encoder,
            "ebill_enrolled_status_encoder": self.ebill_enrolled_status_encoder,
            "card_activation_status_encoder": self.card_activation_status_encoder
        }

        with open("data/"+self.mos_mode+"_encoders.pkl", "wb") as file:
            pickle.dump(encoders, file)

    def save_file(self,df):
        df.to_csv("data/data_preprocessed_"+self.mos_mode+".csv", index = False)
        return df
    
    
    
if __name__ == "__main__":
    for mode in ['head']:#["drop","length","TR","tail","multihot"]
        df = pd.read_csv("data/data.csv")
        pp = preprocessor(debug=True,mode=mode)
        df = pp.process(df)
        pp.save_file(df)
