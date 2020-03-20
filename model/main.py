import os
import numpy as np
from numpy.random import seed

from read_write import read_data, save_model
from feature_extractor import FeatureExtractor
from regressor import Regressor
from scoring import save_scores
from plots import plot_model, plot_history
from settings.dev import TRAIN_FILE, TEST_FILE

seed(42)
np.random.seed(42)
os.environ['PYTHONHASHSEED'] = str(0)


if __name__ == "__main__":
    do_cv = False
    message = " "

    X_train, y_train = read_data(TRAIN_FILE)
    X_test, y_test = read_data(TEST_FILE)

    epoch = 1
    batch = 516
    len_sequences = 10

    feature_ext = FeatureExtractor(len_sequences=len_sequences)
    feature_ext.fit(X_train, y_train)
    X_array = feature_ext.transform(X_train)
    X_array_test = feature_ext.transform(X_test)

    model = Regressor(epochs=epoch, batch=batch,
                      num_scalar=len(feature_ext.scalar_fields),
                      num_const=len(feature_ext.constant_fields),
                      len_sequences=len_sequences)

    history = model.fit(X_array, y_train, do_cv)

    plot_model(model.model)

    save_model(model.model)
    plot_history(history, do_cv)
    save_scores(model, X_array, y_train, X_array_test, y_test, message=message)
