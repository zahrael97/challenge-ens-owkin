import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from utils import _lower


class PandasOneHotEncoder(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        cat_cols=None,
        replace_nan=True,
    ):
        self.cat_cols = cat_cols
        self.mapping = dict()
        self.cols = []

    def fit(self, X, y=None):
        if self.cat_cols is None:
            self.cat_cols_ = list(X.columns)
        else:
            self.cat_cols_ = self.cat_cols

        self.size = 0
        for col in self.cat_cols_:
            X.loc[:, col] = X[col].fillna("__nan__")
            values = sorted(list(_lower(X[col]).unique()))
            self.mapping[col] = {val: i+self.size for i, val in enumerate(values)}
            self.cols += [f"{col}_{val}" for val in values]
            self.size += len(values)
        return self

    def transform(self, X):
        y = np.zeros((X.shape[0], self.size))
        for col in self.cat_cols_:
            X.loc[:, col] = X[col].fillna("__nan__")
            for j, val in enumerate(X[col].values):
                try:
                    y[j, self.mapping[col][_lower(val)]] = 1
                except KeyError:
                    warnings.warn(f"value: {_lower(val)} for column {col} "
                                  "not present in train split")

        y = np.hstack((y, X.PatientID.values.reshape((-1, 1))))
        y = pd.DataFrame(y, columns=self.cols+['PatientID'])
        X = X.drop(columns=self.cat_cols_).copy()
        X = X.merge(y)
        return X


class PandasScaler(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        cont_cols=None,
        handle_nan=True,
    ):
        self.cont_cols = cont_cols
        self.mapping = dict()
        self.cols = []

    def fit(self, X, y=None):
        if self.cont_cols is None:
            self.cont_cols_ = list(X.columns)
        else:
            self.cont_cols_ = self.cont_cols

        for col in self.cont_cols:
            mean = X[col].mean()
            X.loc[:, col] = X[col].fillna(mean)
            self.mapping[col] = {"mean": mean, "std": X[col].std()}
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cont_cols_:
            X[col] = X[col].fillna(self.mapping[col]["mean"])
            X[col] -= self.mapping[col]["mean"]
            X[col] /= self.mapping[col]["std"]
        return X
