import os
import pdb
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from contextlib import redirect_stdout
from shutil import copyfile
from sklearn.metrics import mean_squared_error, r2_score
from feature_extractor import FeatureExtractor
from regressor import Regressor

from numpy.random import seed
seed(42)
np.random.seed(42)
os.environ['PYTHONHASHSEED'] = str(0)


output_path = "output/"
_forecast_h = 24


def _read_data(path, dataset):
    try:
        Data = pd.read_csv(path + '/data/' + dataset + '.csv')
    except IOError:
        raise IOError("Data not found")

    y = np.empty((len(Data)))
    y[:] = np.nan
    # only one instant every 6 h,
    # so the forecast window is 'windowt' timesteps ahead
    windowt = _forecast_h / 6
    for i in range(len(Data)):
        if i + windowt >= len(Data):
            continue
        if Data['instant_t'][i + windowt] - Data['instant_t'][i] == windowt:
            y[i] = Data['windspeed'][i + windowt]
    X = Data
    i_toerase = []
    for i, yi in enumerate(y):
        if math.isnan(yi):
            i_toerase.append(i)
    X = X.drop(X.index[i_toerase])
    X.index = range(len(X))
    y = np.delete(y, i_toerase, axis=0)
    return X, y


def plot_history(path, history, do_cv=False, name="model_loss"):
    keys = list(history.history.keys())
    plt.figure()
    plt.plot(history.history[keys[0]], label="train")
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    if do_cv:
        plt.plot(history.history[keys[1]], label="test")
        plt.title('Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
    plt.legend()
    plt.savefig(path + name + ".png")
    plt.close()


def plot_model(path, model):
    with open(path + 'model.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()


def save_score(path, functions, train_true, train_pred, test_true, test_pred, name="scores",
               message=""):
    f = open(path + name + '.txt', 'w+')
    for function in functions:
        f.write("Scores train " + function.__name__ + ": " +
                str(function(train_true, train_pred)) + "\n")
        f.write("Scores test " + function.__name__ + ": " +
                str(function(test_true, test_pred)) + "\n")
    f.write(message + "\n")
    f.close()


def save_files():
    copyfile('/home/ubuntu/documents/storm_forecast/model/main.py',
             '/home/ubuntu/documents/storm_forecast/model/output/main.py')
    copyfile('/home/ubuntu/documents/storm_forecast/model/regressor.py',
             '/home/ubuntu/documents/storm_forecast/model/output/regressor.py')
    copyfile('/home/ubuntu/documents/storm_forecast/model/feature_extractor.py',
             '/home/ubuntu/documents/storm_forecast/model/output/feature_extractor.py')


def rmse(x, y):
    return np.sqrt(mean_squared_error(x, y))


if __name__ == "__main__":
    do_cv = True
    do_feature_ext = True
    save_feature_ext = False
    message = "attention module"

    X_train, y_train = _read_data("..", "train")
    X_test, y_test = _read_data("..", "test")

    epoch = 200
    len_sequences = 10

    if do_feature_ext:
        feature_ext = FeatureExtractor(len_sequences=len_sequences)
        feature_ext.fit(X_train, y_train)
        X_array = feature_ext.transform(X_train)
        X_array_test = feature_ext.transform(X_test)
        print("Arrays processed")
        if save_feature_ext:
            # np.save("../data/train_norm", X_array[0])
            np.save("../data/train_scalar", X_array[1])
            np.save("../data/train_const", X_array[0])
            # np.save("../data/test_norm", X_array_test[0])
            np.save("../data/test_scalar", X_array_test[1])
            np.save("../data/test_const", X_array_test[0])
    else:
        X_array = [np.load("../data/train_const.npy"), np.load("../data/train_scalar.npy")]
        X_array_test = [np.load("../data/test_const.npy"), np.load("../data/test_scalar.npy")]
    model = Regressor(epochs=epoch, num_scalar=X_array[1].shape[2],
                      num_const=X_array[2].shape[1], len_sequences=len_sequences)
    history = model.fit(X_array, y_train, do_cv)
    pdb.set_trace()
    pred_train = model.predict(X_array)
    pred_test = model.predict(X_array_test)

    plot_model(output_path, model.cnn_model)
    save_files()
    plot_history(output_path, history, do_cv)
    save_score(output_path, [rmse, r2_score], y_train,
               pred_train, y_test, pred_test, message=message)
