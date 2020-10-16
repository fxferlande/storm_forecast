import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from model.scoring import rmse


class Regressor(BaseEstimator):
    def __init__(self, num_scalar=12, num_const=7, len_sequences=5):
        return None

    def fit(self, X: np.ndarray, y: np.ndarray, do_cv: bool = False) -> None:
        return None

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X["windspeed"]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Computes rmse between target y and predictions for input X after
        extracting different subdatasets (image, scalar, and constant).

        Args:
            X  (np.ndarray):     Source array, containing all the type of
                                 inputs, concatenated in one dataframe.
            y    (np.ndarray):   Target

        Returns:
            float:  rmse of predictions
        """
        X = self.extract_subdatasets(X)
        pred = self.model.predict(X)
        return rmse(pred, y)

    def compute_scores(self, X: np.ndarray, y: np.ndarray,
                       name: str = "") -> pd.Series:
        scores = pd.Series()
        pred = self.predict(X)

        scores.loc["RMSE{}".format(name)] = rmse(pred, y)

        return scores
