import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from settings.dev import OUTPUT_DIR


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Simplifies calculation of rmse.
    Args:
        pred  (np.ndarray):   Predicted values
        true  (np.ndarray):   Target values
    Returns:
        float:   RMSE
    """
    if len(pred) != len(true) or len(pred) == 0:
        return None
    return np.sqrt(mean_squared_error(pred, true))


def weighted_rmse(pred: np.ndarray, true: np.ndarray,
                  weights: np.ndarray) -> float:
    """
    Simplifies calculation of weighted rmse.
    Args:
        pred      (np.ndarray):   Predicted values
        true      (np.ndarray):   Target values
        weights   (np.ndarray):   Weights to apply to errors
    Returns:
        float:   RMSE
    """
    if len(pred) != len(true) or len(pred) == 0:
        return None
    return np.sqrt(mean_squared_error(pred, true, sample_weight=weights))


def save_scores(scores: pd.Series, name: str = "/scores",
                message: str = "") -> None:
    """
    Use the compute_scores function on both train and test sets, and saves the
    results in a .txt file.
    Args:
        scores     (pd.Series):   DataFrame with all the scores
        name             (str):   File name
        message          (str):   Message to include in file

    Returns:
        None
    """
    f = open(OUTPUT_DIR + name + '.txt', 'w+')
    for metric in scores.iteritems():
        f.write("{}: {}".format(metric[0], metric[1]) + "\n")
    f.write(message + "\n")
    f.close()
