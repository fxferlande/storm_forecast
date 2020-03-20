import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from settings.dev import OUTPUT_DIR
from model.regressor import Regressor


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Simplifies calculation of rmse.
    Args:
        pred  (np.ndarray):   Predicted values
        true  (np.ndarray):   Target values
    Returns:
        float:   RMSE
    """
    return np.sqrt(mean_squared_error(pred, true))


def custom_rmse(model: Regressor, X: np.ndarray, y: np.ndarray) -> float:
    """
    Allows to weight the RMSE by the length of the sequences. The longer the
    sequence, the higher the weight.
    Args:
        model   (Regressor):   Model used to predict from X
        X      (np.ndarray):   Source array, containing all the type of
                               inputs, concatenated in one dataframe.
        y      (np.ndarray):   Target values
    Returns:
        float:   Custom RMSE
    """
    pred = model.predict(X)
    X_array = model.extract_subdatasets(X)
    len_sequences = np.sum(((X_array[1] > -10)*1)[:, :, 1], axis=1)
    len_sequences = len_sequences/max(len_sequences)
    score = np.sqrt(mean_squared_error(pred, y, sample_weight=len_sequences))
    return score


def rmse_inf(model: Regressor, X: np.ndarray, y: np.ndarray,
             threshold: int) -> float:
    """
    Computes the RMSE only for points where the available history is lower than
    threshold points.
    Args:
        model   (Regressor):   Model used to predict from X
        X      (np.ndarray):   Source array, containing all the type of
                               inputs, concatenated in one dataframe.
        y      (np.ndarray):   Target values
        threshold     (int):   Number limit of historical points
    Returns:
        float:   Custom RMSE
    """
    pred = model.predict(X)
    X_array = model.extract_subdatasets(X)
    len_sequences = np.sum(((X_array[1] > -10)*1)[:, :, 1], axis=1)
    indexes = len_sequences < threshold
    if len(y[indexes]) > 0:
        score = rmse(pred[indexes], y[indexes])
    else:
        score = None
    return score


def rmse_sup(model: Regressor, X: np.ndarray, y: np.ndarray,
             threshold: int) -> float:
    """
    Computes the RMSE only for points where the available history is higher
    than threshold points.
    Args:
        model   (Regressor):   Model used to predict from X
        X      (np.ndarray):   Source array, containing all the type of
                               inputs, concatenated in one dataframe.
        y      (np.ndarray):   Target values
        threshold     (int):   Number limit of historical points
    Returns:
        float:   Custom RMSE
    """
    pred = model.predict(X)
    X_array = model.extract_subdatasets(X)
    len_sequences = np.sum(((X_array[1] > -10)*1)[:, :, 1], axis=1)
    indexes = len_sequences > threshold
    if len(y[indexes]) > 0:
        score = rmse(pred[indexes], y[indexes])
    else:
        score = None
    return score


def compute_scores(model: Regressor, X: np.ndarray,
                   y: np.ndarray) -> pd.DataFrame:
    """
    Computes the different error scores and returns a dataframe with all the
    scores
    Args:
        model   (Regressor):   Model used to predict from X
        X      (np.ndarray):   Source array, containing all the type of
                               inputs, concatenated in one dataframe.
        y      (np.ndarray):   Target values
    Returns:
        pd.DataFrame:   DataFrame with all the scores
    """
    metrics = ["RMSE", "R2", "custom_rmse", "rmse_inf", "rmse_sup"]
    scores = pd.DataFrame(columns=["Valeur"], index=metrics)
    pred = model.predict(X)

    scores.loc["RMSE"] = rmse(pred, y)
    scores.loc["R2"] = r2_score(pred, y)
    scores.loc["custom_rmse"] = custom_rmse(model, X, y)
    scores.loc["rmse_inf"] = rmse_inf(model, X, y, 5)
    scores.loc["rmse_sup"] = rmse_sup(model, X, y, 5)

    return scores


def save_scores(model: Regressor, train: np.ndarray, y_train: np.ndarray,
                test: np.ndarray, y_test: np.ndarray, name: str = "/scores",
                message: str = "") -> None:
    """
    Use the compute_scores function on both train and test sets, and saves the
    results in a .txt file.
    Args:
        model    (Regressor):   Model used to predict from train and test
        train   (np.ndarray):   Source array, containing all the type of
                                inputs, concatenated in one dataframe for train
        y_train (np.ndarray):   Target values for train
        test    (np.ndarray):   Source array, containing all the type of
                                inputs, concatenated in one dataframe for test
        y_test  (np.ndarray):   Target values for test
        name           (str):   File name
        message        (str):   Message to include in file

    Returns:
        None
    """
    scores_train = compute_scores(model, train, y_train)
    scores_test = compute_scores(model, test, y_test)
    f = open(OUTPUT_DIR + name + '.txt', 'w+')
    f.write("Scores train \n")
    for metric in scores_train.iterrows():
        f.write(str(metric[0]) + ": " + str(metric[1].Valeur) + "\n")
    f.write("\nScores test \n")
    for metric in scores_test.iterrows():
        f.write(str(metric[0]) + ": " + str(metric[1].Valeur) + "\n")
    f.write(message + "\n")
    f.close()
