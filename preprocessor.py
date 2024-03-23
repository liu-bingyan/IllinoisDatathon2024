import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

class preprocessor():
    def __init__(self):
        pass
        self.les = []
        self.num_idx = []
        self.cat_idx = []
        self.cat_dim = []

    def process(self,df):
        df = self.process_columns(df)
        df = self.encode_labels(df)
        self.classify_columns(df)
        return df

    def process_columns(self,df):
        df = self.process_mos_rough(df)
        #df = self.process_mos_one_hot(df)
        df = self.process_timestamp_call_key(df)
        #df = self.process_account_open_date(df,day = 13)
        #df = self.process_account_open_date(df,day = 18)
        df = self.process_resolved(df)

        print("new columns : ",df.columns)
        return df  
    
    #.....
    def encode_labels(self,df):
        for col in df.columns:
            le = LabelEncoder()
            self.les.append(le)
            df[col] = le.fit_transform(df[col])
        return df

    def classify_columns(self,df):
        # collect index of categorical columns
        # collect dimension of each categorical columns
        # collect index of numerical columns
        self.num_idx = []
        self.cat_idx = []
        self.cat_dim = []
    
    ############################## helper functions ########################################
    def process_resolved(self,df):
        df['resolved']= df['resolved'].apply(lambda x : 1 if x =='resolved' else 0)
        return df
    
    def process_mos_rough(self,df):
        df['mos_count'] = df['mos'].apply(lambda x: len(x.split(" ")))
        df['mos_TR'] = df['mos'].apply(lambda x: 1 if 'TR' in x else 0)
        return df 
    
    # to be implemented
    def process_mos_one_hot(self,df):
        print('hellow world!')
        pass

    def process_timestamp_call_key(self, df):
        df["timestamp"] = (df["timestamp_call_key"].str.slice(start = 0, stop = 8))
        df["timestamp_year"] = (df["timestamp_call_key"].str.slice(start = 0, stop = 2))
        df["timestamp_month"] = (df["timestamp_call_key"].str.slice(start = 2, stop = 4))
        df["timestamp_day"] = (df["timestamp_call_key"].str.slice(start = 4, stop = 6))
        df["timestamp_hour"] = (df["timestamp_call_key"].str.slice(start = 6, stop = 8))
        df["timestamp_min"] = (df["timestamp_call_key"].str.slice(start = 8, stop = 10))
        df["call_key"] = (df["timestamp_call_key"].str.slice(start = 11, stop = 21))
        return df
    
    def process_account_open_date(self, df, day = 13):
        df['account_open_year_'+str(day)] = df['account_open_date_'+str(day)+'_march'].dt.year
        df['account_open_month_'+str(day)] = df['account_open_date_'+str(day)+'_march'].dt.month
        df['account_open_day_'+str(day)] = df['account_open_date_'+str(day)+'_march'].dt.day
        df["account_age_years_"+str(day)] = 2024 - df['account_open_date_'+str(day)+'_march'].dt.year
        return df
    

if __name__ == "__main__":
    # = pd.read_csv("data.csv")
    pc = preprocessor()
    df,num_idx,cat_idx,cat_dim = pc.process(df)