import math
import pandas as pd
import numpy as np

from settings.dev import DATA_DIR, OUTPUT_DIR


def read_data(dataset: str, horizon: int = 24) -> (pd.DataFrame, np.ndarray):
    """
    Reads data from the DATA_DIR and separates into target and features. Target
    is computed by selecting windspeed feature and selecting value of the
    same storm at a certain horizon ahead.
    Args:
        dataset   (str):   Name of the dataset in DATA_DIR
        horizon   (int):   Horizon for target (in hours)

    Returns:
        (pd.DataFrame, np.ndarray) :   Tuple of features and target
    """
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


def save_model(model, name: str = 'model') -> None:
    # unused
    """
    Saves the keras model in .h5 format in OUTPUT_DIR.
    Args:
        name   (str):   Name of the output file

    Returns:
        None
    """
    model.save(OUTPUT_DIR + '/{}.h5'.format(name))
