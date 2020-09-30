import logging
import time
import numpy as np
import pandas as pd
import keras.backend as K
from sklearn.base import BaseEstimator
from keras.layers import Concatenate, Dropout, Activation, Dense, Input, \
     Flatten, Conv1D, LSTM, Permute, RepeatVector, Multiply, Lambda
from keras.models import Model
from keras.regularizers import l2
from keras.callbacks.callbacks import History
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from model.scoring import rmse, weighted_rmse


class Regressor(BaseEstimator):
    def __init__(self, num_scalar=12, num_const=7, len_sequences=5):
        self.epochs = 800
        self.batch = 2000
        self.len_lstm = 32
        self.len_conv = 128
        self.l2_weight = 4e-5
        self.l2_lstm = 3e-5
        self.l2_conv = 3e-2

        self.lr = 0.00005

        self.len_sequences = len_sequences
        self.num_scalar = num_scalar
        self.num_const = num_const

        self.init_model()

    def init_model(self, verbose: int = 1) -> None:
        """
        Creates the keras model for training. It is based on 4 blocks
        corresponding to 4 sub-models:
        - one processing the images
        - one processing the scalar sequences through LSTM
        - one processing the scalar sequences through CNN
        - one processing the constant features
        They are all concatenated at the end.

        Args:
            verbose   (int):   Option to show the parameters of the model once
                               compiled

        Returns:
            None
        """

        scalar_in = Input(shape=(self.len_sequences, self.num_scalar))
        const_in = Input(shape=(self.num_const,))

        model_scalar = scalar_in
        activations = LSTM(self.len_lstm, activation='relu',
                           kernel_regularizer=l2(self.l2_lstm),
                           return_sequences=True)(model_scalar)
        attention = Dense(1, activation='relu')(activations)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(self.len_lstm)(attention)
        attention = Permute([2, 1])(attention)

        sent_representation = Multiply()([activations, attention])
        sent_representation = Lambda(lambda xin: K.sum(xin, axis=-2),
                                     output_shape=(self.len_lstm,))(
                                     sent_representation)
        model_scalar = Dense(256)(sent_representation)
        model_scalar = Dropout(0.1)(model_scalar)
        model_scalar = Dense(64)(model_scalar)
        model_scalar = Dense(32)(model_scalar)
        model_scalar = Activation("tanh")(model_scalar)

        model_scalar_2 = Conv1D(self.len_conv, 3, padding="same",
                                kernel_regularizer=l2(3e-5)
                                )(scalar_in)
        model_scalar_2 = Activation("selu")(model_scalar_2)
        model_scalar_2 = Flatten()(model_scalar_2)
        model_scalar_2 = Dense(32, kernel_regularizer=l2(self.l2_weight))(
            model_scalar_2)
        model_scalar_2 = Dense(16, kernel_regularizer=l2(self.l2_weight))(
            model_scalar_2)
        model_scalar_2 = Dropout(0.2)(model_scalar_2)
        model_scalar_2 = Activation("tanh")(model_scalar_2)

        model_const = const_in
        model_const = Dense(64, kernel_regularizer=l2(self.l2_weight))(
            model_const)
        model_const = Activation("tanh")(model_const)

        model = Concatenate()([model_scalar, model_scalar_2,
                               model_const])

        model = Dense(128, kernel_regularizer=l2(self.l2_weight))(model)
        model = Dense(64, kernel_regularizer=l2(self.l2_weight))(model)
        model = Activation("tanh")(model)
        model = Dropout(0.2)(model)
        model = Dense(32, kernel_regularizer=l2(self.l2_weight))(model)
        model = Activation("tanh")(model)

        model = Dense(1)(model)

        self.model = Model([scalar_in, const_in], model)
        self.model.compile(loss="mse", optimizer=Adam(learning_rate=self.lr))

        if verbose == 1:
            print(self.model.summary())
        return

    def fit(self, X: np.ndarray, y: np.ndarray, do_cv: bool = False,
            verbose: int = 1) -> History:
        """
        Fits the model on X and y after extracting different subdatasets
        (image, scalar, and constant) from X.

        Args:
            X   (np.ndarray):    Source array, containing all the type of
                                 inputs, concatenated in one dataframe.
            y    (np.ndarray):   Target
            do_cv      (bool):   Option for keras integrated cross-validation
            verbose     (int):   Verbose parameter for keras fit function

        Returns:
            History:  Keras history of training (useful for plotting curves)
        """
        t = time.time()
        X = self.extract_subdatasets(X)

        indexes = np.sum((X[0][:, -1] == -100)*1, axis=1)  \
            <= 10
        X = [x[indexes] for x in X]
        x, _ = X

        self.target_mean = y.mean()
        self.target_std = y.std()
        y = (y - self.target_mean)/self.target_std
        y = y[indexes] - x[:, self.len_sequences-1, 1]
        if do_cv:
            callback = EarlyStopping(monitor='val_loss', min_delta=0.01,
                                     patience=100)
            history = self.model.fit(X, y, epochs=self.epochs,
                                     batch_size=self.batch,
                                     verbose=verbose,
                                     validation_split=0.2,
                                     callbacks=[callback])
        else:
            callback = EarlyStopping(monitor='loss', min_delta=0.01,
                                     patience=50)
            history = self.model.fit(X, y, epochs=self.epochs,
                                     batch_size=self.batch,
                                     verbose=verbose,
                                     callbacks=[callback])
        duration = int((time.time()-t)/60)
        logging.info("Training done in {:.0f} mins".format(duration))
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts output for X after extracting different subdatasets
        (image, scalar, and constant).

        Args:
            X  (np.ndarray):    Source array, containing all the type of
                                Inputs, concatenated in one dataframe.

        Returns:
            np.ndarray:  Predicted values for windspeed
        """
        X = self.extract_subdatasets(X)
        x, _ = X
        pred = self.model.predict(X).ravel() + \
            x[:, self.len_sequences-1, 1]
        pred = pred*self.target_std + self.target_mean
        return pred

    def extract_subdatasets(self, X: np.ndarray) -> list:
        """
        Extracts different subdatasets from X (image, scalar, and constant).
        Based on the expected shapes, it splits X in three subdatasets and
        reshapes them. Il allows us to give a DataFrame as input for the model.

        Args:
            X  (np.ndarray):    source array, containing all the type of
                                inputs, concatenated in one dataframe.

        Returns:
            list:  List of the 3 subdatasets
        """
        break_scalar = self.len_sequences*self.num_scalar
        break_const = break_scalar + self.num_const
        num_samples = len(X)

        shape_scalar = (num_samples, self.len_sequences, self.num_scalar)

        norm_scalar = np.reshape(X[:, :break_scalar], shape_scalar)
        norm_const = X[:, break_scalar: break_const]

        return [norm_scalar, norm_const]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Computes rmse between target y and predictions for input X after
        extracting different subdatasets (image, scalar, and constant).

        Args:
            X  (np.ndarray):     Source array, containing all the type of
                                 inputs, concatenated in one dataframe.
            y    (np.ndarray):   Target

        Returns:
            float:  rmse of predictions
        """
        X = self.extract_subdatasets(X)
        pred = self.model.predict(X)
        return rmse(pred, y)

    def compute_scores(self, X: np.ndarray, y: np.ndarray,
                       name: str = "") -> pd.Series:
        scores = pd.Series()
        pred = self.predict(X)
        X_array = self.extract_subdatasets(X)
        len_sequences = np.sum(((X_array[0] > -10)*1)[:, 1], axis=1)

        scores.loc["RMSE{}".format(name)] = rmse(pred, y)

        weight = len_sequences/max(len_sequences)
        scores.loc["cust_rmse{}".format(name)] = weighted_rmse(pred, y, weight)

        return scores

    def get_params(self, deep: bool = True) -> dict:
        """
        Returns the parameters of the model.

        Args:
            deep  (bool):  True

        Returns:
            dict:   Parameters of the model
        """
        return {"epochs": self.epochs,
                "batch": self.batch,
                "len_sequences": self.len_sequences,
                "num_scalar": self.num_scalar,
                "num_const": self.num_const}

    def set_params(self, **parameters) -> None:
        """
        Sets the parameters of the model.

        Returns:
            None
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        self.init_model(verbose=0)
        return self
