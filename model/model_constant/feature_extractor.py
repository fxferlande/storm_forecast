import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment', None)


class FeatureExtractor(object):
    def __init__(self, len_sequences: int = 10):
        self.constant_fields = []
        self.scalar_fields = []
        return None

    def fit(self, X_df: pd.DataFrame, y: np.ndarray = None) -> None:
        return None

    def transform(self, X_df: pd.DataFrame) -> np.ndarray:
        return X_df
