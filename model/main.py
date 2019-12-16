import os
import numpy as np
from numpy.random import seed

from read_write import read_data, save_files, save_model
from feature_extractor import FeatureExtractor
from regressor import Regressor
from scoring import save_scores
from plots import plot_model, plot_history

seed(42)
np.random.seed(42)
os.environ['PYTHONHASHSEED'] = str(0)
output_path = "output/"


if __name__ == "__main__":
    do_cv = False
    message = " "

    X_train, y_train = read_data(".", "train")
    X_test, y_test = read_data(".", "test")

    epoch = 1000
    batch = 516
    len_sequences = 5

    feature_ext = FeatureExtractor(len_sequences=len_sequences)
    feature_ext.fit(X_train, y_train)
    X_array = feature_ext.transform(X_train)
    X_array_test = feature_ext.transform(X_test)

    model = Regressor(epochs=epoch, batch=batch,
                      num_scalar=len(feature_ext.scalar_fields),
                      num_const=len(feature_ext.constant_fields),
                      len_sequences=len_sequences)

    history = model.fit(X_array, y_train, do_cv)

    plot_model(output_path, model.cnn_model)
    save_files()
    save_model(output_path, model.cnn_model)
    plot_history(output_path, history, do_cv)
    save_scores(output_path, model, X_array, y_train, X_array_test, y_test,
                message=message)
