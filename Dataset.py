import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np


class StaticDataset:
    def __init__(self,feature_cols: list = None, target_col: str = None, max_rows: int = 1000000, min_rows: int = 10000):#NEED TO INPUT TARGET COL
        self.raw_df = None
        self.df = None
        self.scaler = StandardScaler()
        self.feature_cols = feature_cols# Would be nice to have inputed, but optional
        self.target_col = target_col
        self.train_df = None
        self.X = None
        self.Y = None
        self.MAX_ROWS = max_rows
        self.MIN_ROWS = min_rows

    def input_dataset(self,location):
        df = location
        self.raw_df = df

        # if no features col, use every thing but target
        if self.feature_cols is None:
            if self.target_col:
                self.feature_cols = [col for col in self.raw_df.columns if col != self.target_col]
            else:
                self.feature_cols = self.raw_df.columns.tolist()


    def preprocess(self):
        self._clean()
        self._update_feature_cols()
        self._allocate_training_size()
        self._normalise()
        self._split_XY()
        return self # not sure what this does yet

    def _clean(self):# copied from main
        df = self.raw_df.copy()

        #Remove unnamed/index columns
        unnamed_cols = [col for col in df.columns if 'unnamed' in col.lower() or 'index' in col.lower()]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)

        #Remove all empty row/columns
        df = df.dropna(how='all', axis=0)  # Drop rows
        df = df.dropna(how='all', axis=1)  # Drop columns

        #Remove duplicates
        df = df.drop_duplicates()

        #Drop rows where >50% of values are missing
        threshold = len(df.columns) * 0.5
        df = df.dropna(thresh=threshold, axis=0)

        self.df = df

    def _update_feature_cols(self):
        if self.feature_cols:
            self.feature_cols = [col for col in self.feature_cols if col in self.df.columns]
        if self.target_col and self.target_col in self.feature_cols:
            self.feature_cols.remove(self.target_col)

    def _allocate_training_size(self):
        n = len(self.df)
        if n <= self.MAX_ROWS:
            self.train_df = self.df.copy()
        else:
            ten_percent = int(n * 0.10)

            if ten_percent >= self.MIN_ROWS:
                self.train_df = self.df.sample(ten_percent, random_state=42)
            else:
                # 10% is too small, use the minimum needed instead
                self.train_df = self.df.sample(self.MIN_ROWS, random_state=42)

    def _normalise(self):
        features = self.df[self.feature_cols]
        numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()#which columns are numbers and which arent
        categorical_cols = features.select_dtypes(exclude=[np.number]).columns.tolist()

        self.preprocessor = ColumnTransformer(transformers=[
            ("num", StandardScaler(), numeric_cols),#normalises all numbers so that different columns dont out weight eachother
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),]) #one hot encodes all categories so that all items in a category are weighted the same
        self.preprocessor.fit(self.train_df[self.feature_cols])  # learns from the data, so that we can transform it.

    def _split_XY(self):
        transformed = self.preprocessor.transform(self.df[self.feature_cols])# transforms the normalised data
        self.X = transformed

        if self.target_col:
            self.Y = self.df[self.target_col].values
            print(f"[split] X: {self.X.shape} | Y: {self.Y.shape}")
        else:
            print(f"[split] X: {self.X.shape} | no target column set")
