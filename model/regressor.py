import logging
import time
import numpy as np
import pandas as pd
import keras.backend as K
from sklearn.base import BaseEstimator
from keras.layers import Concatenate, Dropout, Activation, Dense, Input, \
     Flatten, Conv2D, Conv1D, MaxPooling2D, LSTM, Permute, RepeatVector, \
     Multiply, Lambda, Add, LeakyReLU
from keras.models import Model
from keras.regularizers import l2
from keras.callbacks.callbacks import History
from model.scoring import rmse, weighted_rmse


class Regressor(BaseEstimator):
    def __init__(self, num_scalar=12, num_const=7, epochs=200,
                 batch=128, len_sequences=5, len_lstm=4):
        self.epochs = epochs
        self.batch = batch
        self.len_sequences = len_sequences
        self.len_lstm = len_lstm
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
        l2_weight = 1
        l2_lstm = 10
        l2_conv = 10

        img_in = Input(shape=(11, 11, 7))
        scalar_in = Input(shape=(self.len_sequences, self.num_scalar))
        const_in = Input(shape=(self.num_const,))

        model_img = img_in
        model_img = Conv2D(32, (5, 5), padding="same",
                           kernel_regularizer=l2(l2_conv))(model_img)
        model_img = LeakyReLU(alpha=0.5)(model_img)

        img_shortcut = model_img
        model_img = Conv2D(32, (5, 5), padding="same",
                           kernel_regularizer=l2(l2_conv))(model_img)
        model_img = LeakyReLU(alpha=0.5)(model_img)

        model_img = Conv2D(32, (5, 5), padding="same",
                           kernel_regularizer=l2(l2_conv))(model_img)
        model_img = Add()([model_img, img_shortcut])
        model_img = LeakyReLU(alpha=0.5)(model_img)
        model_img = MaxPooling2D()(model_img)

        model_img = Conv2D(64, (3, 3), padding="same",
                           kernel_regularizer=l2(l2_conv))(model_img)
        model_img = LeakyReLU(alpha=0.5)(model_img)

        img_shortcut = model_img
        model_img = Conv2D(64, (3, 3), padding="same",
                           kernel_regularizer=l2(l2_conv))(model_img)
        model_img = LeakyReLU(alpha=0.5)(model_img)

        model_img = Conv2D(64, (3, 3), padding="same",
                           kernel_regularizer=l2(l2_conv))(model_img)
        model_img = Add()([model_img, img_shortcut])
        model_img = LeakyReLU(alpha=0.5)(model_img)
        model_img = MaxPooling2D()(model_img)

        model_img = Conv2D(128, (3, 3), padding="same",
                           kernel_regularizer=l2(l2_conv))(model_img)
        model_img = LeakyReLU(alpha=0.5)(model_img)

        img_shortcut = model_img
        model_img = Conv2D(128, (3, 3), padding="same",
                           kernel_regularizer=l2(l2_conv))(model_img)
        model_img = LeakyReLU(alpha=0.5)(model_img)

        model_img = Conv2D(128, (3, 3), padding="same",
                           kernel_regularizer=l2(l2_conv))(model_img)
        model_img = Add()([model_img, img_shortcut])
        model_img = LeakyReLU(alpha=0.5)(model_img)
        model_img = MaxPooling2D()(model_img)

        model_img = Flatten()(model_img)

        model_img = Dense(128, kernel_regularizer=l2(l2_weight))(model_img)
        model_img = Dropout(0.15)(model_img)
        model_img = Activation("tanh")(model_img)

        model_scalar = scalar_in
        activations = LSTM(self.len_lstm, activation='tanh',
                           kernel_regularizer=l2(l2_lstm),
                           return_sequences=True)(model_scalar)
        attention = Dense(1, activation='tanh')(activations)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(self.len_lstm)(attention)
        attention = Permute([2, 1])(attention)

        sent_representation = Multiply()([activations, attention])
        sent_representation = Lambda(lambda xin: K.sum(xin, axis=-2),
                                     output_shape=(self.len_lstm,))(
                                     sent_representation)
        model_scalar = Dense(128)(sent_representation)
        model_scalar = Dense(64)(sent_representation)
        model_scalar = Activation("tanh")(model_scalar)

        model_scalar_2 = Conv1D(64, 3, padding="same")(scalar_in)
        model_scalar_2 = Activation("selu")(model_scalar_2)
        model_scalar_2 = Flatten()(model_scalar_2)
        model_scalar_2 = Dense(16, kernel_regularizer=l2(l2_weight))(
            model_scalar_2)
        model_scalar_2 = Dropout(0.3)(model_scalar_2)
        model_scalar_2 = Activation("tanh")(model_scalar_2)

        model_const = const_in
        model_const = Dense(64, kernel_regularizer=l2(l2_weight))(model_const)
        model_const = Activation("tanh")(model_const)

        model = Concatenate()([model_img, model_scalar, model_scalar_2,
                               model_const])

        model = Dense(64, kernel_regularizer=l2(l2_weight))(model)
        model = Dropout(0.3)(model)
        model = Activation("tanh")(model)

        model = Dense(1)(model)

        self.model = Model([img_in, scalar_in, const_in], model)
        self.model.compile(loss="mse", optimizer="adam")

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
        _, x, _ = X
        y = y - x[:, self.len_sequences-1, 1]
        if do_cv:
            history = self.model.fit(X, y, epochs=self.epochs,
                                     batch_size=self.batch,
                                     verbose=verbose,
                                     validation_split=0.2)
        else:
            history = self.model.fit(X, y, epochs=self.epochs,
                                     batch_size=self.batch,
                                     verbose=verbose)
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
        _, x, _ = X
        pred = self.model.predict(X).ravel() + \
            x[:, self.len_sequences-1, 1]
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
        break_images = 11*11*7
        break_scalar = break_images + self.len_sequences*self.num_scalar
        break_const = break_scalar + self.num_const
        num_samples = len(X)

        shape_image = (num_samples, 11, 11, 7)
        shape_scalar = (num_samples, self.len_sequences, self.num_scalar)

        norm_images = np.reshape(X[:, :break_images], shape_image)
        norm_scalar = np.reshape(X[:, break_images:break_scalar], shape_scalar)
        norm_const = X[:, break_scalar: break_const]

        return [norm_images, norm_scalar, norm_const]

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
        len_sequences = np.sum(((X_array[1] > -10)*1)[:, :, 1], axis=1)

        scores.loc["RMSE{}".format(name)] = rmse(pred, y)

        weight = len_sequences/max(len_sequences)
        scores.loc["cust_rmse{}".format(name)] = weighted_rmse(pred, y, weight)

        idx_inf = len_sequences < 5
        scores.loc["rmse_inf{}".format(name)] = rmse(pred[idx_inf], [idx_inf])

        idx_sup = len_sequences >= 5
        scores.loc["rmse_sup{}".format(name)] = rmse(pred[idx_sup], [idx_sup])

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
