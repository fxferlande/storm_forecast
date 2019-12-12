import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


def rmse(x, y):
    return np.sqrt(mean_squared_error(x, y))


def custom_rmse(model, X, y):
    pred = model.predict(X)
    X_array = model.extract_subdatasets(X)
    len_sequences = np.sum(((X_array[1] > -10)*1)[:, :, 1], axis=1)
    len_sequences = len_sequences/max(len_sequences)
    score = np.sqrt(mean_squared_error(pred, y, sample_weight=len_sequences))
    return score


def rmse_inf(model, X, y, threshold):
    pred = model.predict(X)
    X_array = model.extract_subdatasets(X)
    len_sequences = np.sum(((X_array[1] > -10)*1)[:, :, 1], axis=1)
    indexes = len_sequences < threshold
    if len(y[indexes]) > 0:
        score = rmse(pred[indexes], y[indexes])
    else:
        score = None
    return score


def rmse_sup(model, X, y, threshold):
    pred = model.predict(X)
    X_array = model.extract_subdatasets(X)
    len_sequences = np.sum(((X_array[1] > -10)*1)[:, :, 1], axis=1)
    indexes = len_sequences > threshold
    if len(y[indexes]) > 0:
        score = rmse(pred[indexes], y[indexes])
    else:
        score = None
    return score


def compute_scores(model, X, y):
    metrics = ["RMSE", "R2", "custom_rmse", "rmse_inf", "rmse_sup"]
    scores = pd.DataFrame(columns=["Valeur"], index=metrics)
    pred = model.predict(X)
    scores.loc["RMSE"] = rmse(pred, y)
    scores.loc["R2"] = r2_score(pred, y)
    scores.loc["custom_rmse"] = custom_rmse(model, X, y)
    scores.loc["rmse_inf"] = rmse_inf(model, X, y, 5)
    scores.loc["rmse_sup"] = rmse_sup(model, X, y, 5)

    return scores


def save_scores(path, model, train, y_train, test, y_test, name="scores",
                message=""):
    scores_train = compute_scores(model, train, y_train)
    scores_test = compute_scores(model, test, y_test)
    f = open(path + name + '.txt', 'w+')
    f.write("Scores train \n")
    for metric in scores_train.iterrows():
        f.write(str(metric[0]) + ": " + str(metric[1].Valeur) + "\n")
    f.write("\nScores test \n")
    for metric in scores_test.iterrows():
        f.write(str(metric[0]) + ": " + str(metric[1].Valeur) + "\n")
    f.write(message + "\n")
    f.close()
