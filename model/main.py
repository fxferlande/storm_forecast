import os
import math
import pandas as pd
import numpy as np

from shutil import copyfile
from feature_extractor import FeatureExtractor
from regressor import Regressor
from scoring import save_scores
from plots import plot_model, plot_history

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


def save_files():
    dir = '/home/ubuntu/documents/storm_forecast'
    copyfile(dir + '/model/main.py', dir + '/output/main.py')
    copyfile(dir + '/model/regressor.py', dir + '/output/regressor.py')
    copyfile(dir + '/model/feature_extractor.py',
             dir + '/output/feature_extractor.py')


def save_model(path, model, name='model'):
    model.save(path + '{}.h5'.format(name))


if __name__ == "__main__":
    do_cv = False
    message = " "

    X_train, y_train = _read_data(".", "train")
    X_test, y_test = _read_data(".", "test")

    epoch = 200
    batch = 128
    len_sequences = 10

    feature_ext = FeatureExtractor(len_sequences=len_sequences)
    feature_ext.fit(X_train, y_train)
    X_array = feature_ext.transform(X_train)
    X_array_test = feature_ext.transform(X_test)

    model = Regressor(epochs=epoch, batch=batch,
                      num_scalar=X_array[1].shape[2],
                      num_const=X_array[2].shape[1],
                      len_sequences=len_sequences)
    history = model.fit(X_array, y_train, do_cv)

    plot_model(output_path, model.cnn_model)
    # save_files()
    save_model(output_path, model.cnn_model)
    plot_history(output_path, history, do_cv)
    save_scores(output_path, model, X_array, y_train, X_array_test, y_test,
                message=message)
