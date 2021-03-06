import os
import numpy as np
import pandas as pd
from numpy.random import seed

from model.read_write import read_data
from model.model_global.feature_extractor import FeatureExtractor
from model.model_global.regressor import Regressor
from model.scoring import save_scores
from model.plots import plot_history
from settings.dev import TRAIN_FILE, TEST_FILE

seed(42)
np.random.seed(42)
os.environ['PYTHONHASHSEED'] = str(0)
os.environ['TF_DETERMINISTIC_OPS'] = '1'


if __name__ == "__main__":
    do_cv = False
    message = " "

    len_sequences = 5
    pred_horizon = 24
    max_padding = len_sequences

    X_train, y_train = read_data(TRAIN_FILE, horizon=pred_horizon)
    X_test, y_test = read_data(TEST_FILE, horizon=pred_horizon)

    feature_ext = FeatureExtractor(len_sequences=len_sequences)
    feature_ext.fit(X_train, y_train)
    X_array = feature_ext.transform(X_train)
    X_array_test = feature_ext.transform(X_test)

    X_array = feature_ext.restrict_sequences(X_array, max_padding=max_padding)

    model = Regressor(num_scalar=len(feature_ext.scalar_fields),
                      num_const=len(feature_ext.constant_fields) +
                      feature_ext.num_dummy,
                      len_sequences=len_sequences)

    history = model.fit(X_array, y_train, do_cv)

    plot_history(history, do_cv)
    scores_train = model.compute_scores(X_array, y_train, name="_train")
    scores_test = model.compute_scores(X_array_test, y_test, name="_test")

    scores = pd.concat([scores_train, scores_test], axis=0)
    print(scores)
    save_scores(scores)
