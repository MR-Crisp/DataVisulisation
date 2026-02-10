import pandas as pd

def input_dataset(location):
    df = pd.read_csv(location)
    print(df.head())
    print(df.columns)
    print(df.shape)

testset = "./taxi_zone_lookup.csv"

input_dataset(testset)