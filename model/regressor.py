import logging
import time
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from model.scoring import rmse

PARAMS = dict(n_estimators=1000,
              max_depth=5,
              max_features=10)


class Regressor(BaseEstimator):
    def __init__(self):
        self.init_model()

    def init_model(self, verbose: int = 1) -> None:
        """
        Creates the Random Forest model.
        Returns:
            None
        """
        self.model = RandomForestRegressor(**PARAMS)

    def fit(self, X: np.ndarray, y: np.ndarray, do_cv: bool = False,
            verbose: int = 1) -> None:
        """
        Fits the model on X and y after extracting different subdatasets
        (image, scalar, and constant) from X.

        Args:
            X   (np.ndarray):    Source array, containing all the type of
                                 inputs, concatenated in one dataframe.
            y    (np.ndarray):   Target
            do_cv      (bool):   Option for keras integrated cross-validation
            verbose     (int):   Verbose parameter for keras fit function

        Returns:
            History:  Keras history of training (useful for plotting curves)
        """
        t = time.time()
        y = y - X.windspeed
        self.model.fit(X, y)

        duration = int((time.time()-t)/60)
        logging.info("Training done in {:.0f} mins".format(duration))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts output for X after extracting different subdatasets
        (image, scalar, and constant).

        Args:
            X  (np.ndarray):    Source array, containing all the type of
                                Inputs, concatenated in one dataframe.

        Returns:
            np.ndarray:  Predicted values for windspeed
        """
        pred = self.model.predict(X).ravel() + X.windspeed
        return pred

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

    def set_params(self, **parameters) -> None:
        """
        Sets the parameters of the model.

        Returns:
            None
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        self.init_model(verbose=0)
        return self
