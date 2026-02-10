import pandas as pd

class StaticDataset:
    def __init__(self):
        self.df = None

    def input_dataset(self,location):
        df = pd.read_csv(location)
        self.df = df
        # print(df.head())
        # print(df.columns)
        # print(df.shape)
        print(df.isnull().sum())

    def clean_dataset(self):
        df = self.df.copy()
        #remove all empty row/col
        df = df.dropna(how='all',axis=0)
        df = df.dropna(how='all',axis=1)
        #remove duplicate records
        df = df.drop_duplicates()
        # Drop rows where >50% of values are missing
        threshold = len(df.columns) * 0.5
        df = df.dropna(thresh=threshold, axis=0)
        self.df = df
        print(df.isnull().sum())


testset = "./taxi_zone_lookup.csv"

D = StaticDataset()
D.input_dataset(testset)
D.clean_dataset()
print(D.df.shape)