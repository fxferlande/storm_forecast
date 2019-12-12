import math
import pandas as pd
import numpy as np
from shutil import copyfile

_forecast_h = 24


def read_data(path, dataset):
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
