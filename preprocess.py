class preprocess():
    def __init__(self):
        pass

    def process(self,df):
        df = self.process_mos_rough(df)
        return df   
    
    def process_mos_rough(self,df):
        df['mos_count'] = df['mos'].apply(lambda x: len(x.split(" ")))
        df['mos_TR'] = df['mos'].apply(lambda x: 1 if 'TR' in x else 0)
        return df 
    
    def process_mos_one_hot(self,df):
        pass