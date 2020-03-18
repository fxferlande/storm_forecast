import math
import pandas as pd
import numpy as np

from settings.dev import DATA_DIR, OUTPUT_DIR


def read_data(dataset: str, horizon: int = 24) -> (pd.DataFrame, np.ndarray):
    try:
        Data = pd.read_csv(DATA_DIR + dataset)
    except IOError:
        raise IOError("Data not found")

    y = np.empty((len(Data)))
    y[:] = np.nan

    windowt = horizon / 6
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


def save_model(model, name='model'):
    model.save(OUTPUT_DIR + '/{}.h5'.format(name))
