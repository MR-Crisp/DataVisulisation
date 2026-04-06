import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class StaticDataset:
    def __init__(self,feature_cols: list, target_col: str = None):
        self.df = None
        self.scaler = StandardScaler()
        self.feature_col = feature_cols
        self.target_col = target_col
        self.train_df = None
        self.X = None
        self.Y = None

    def preprocess(self):
        self._clean()
        self._allocate_training_size()
        self._normalise()
        self._split_XY()
        return self # not sure what this does yet

    