from keras.layers import Concatenate, Dropout, BatchNormalization, Conv3D, Activation, Dense, Input
from keras.layers import MaxPooling3D, Flatten, LSTM
from keras.models import Model
from keras.regularizers import l2
from sklearn.base import BaseEstimator


class Regressor(BaseEstimator):
    def __init__(self, epochs=200, len_sequences=5):
        self.epochs = epochs
        self.len_sequences = len_sequences
        l2_weight = 0.1  # 0.0001
        model_in = Input(shape=(len_sequences, 11, 11, 7))
        scalar_in = Input(shape=(len_sequences, 9))

        model = BatchNormalization()(model_in)
        model = Conv3D(64, (5, 5, 5), padding="same")(model)
        model = Activation("relu")(model)
        model = MaxPooling3D()(model)

        model = BatchNormalization()(model)
        model = Conv3D(128, (5, 5, 5), padding="same")(model)
        model = Activation("relu")(model)

        model = MaxPooling3D()(model)
        model = BatchNormalization()(model)
        model = Conv3D(128, (3, 3, 3), padding="same")(model)
        model = Activation("relu")(model)
        model = Flatten()(model)

        model = Dense(128, kernel_regularizer=l2(l2_weight))(model)
        model = Activation("tanh")(model)

        model_scalar = LSTM(30, activation='tanh',
                            kernel_regularizer=l2(l2_weight))(scalar_in)
        model_scalar = Dense(128)(model_scalar)

        model = Concatenate()([model, model_scalar, model_scalar, model_scalar])
        model = BatchNormalization()(model)

        model = Dense(128, kernel_regularizer=l2(l2_weight))(model)
        model = Activation("tanh")(model)

        model = Dense(32, kernel_regularizer=l2(l2_weight))(model)
        model = Dropout(0.5)(model)
        model = Activation("tanh")(model)

        model = Dense(1)(model)

        self.cnn_model = Model([model_in, scalar_in], model)
        self.cnn_model.compile(loss="mse", optimizer="adam")

        print(self.cnn_model.summary())
        return

    def fit(self, X, y, do_cv=False):
        _, x = X
        y = y - x[:, self.len_sequences-1, 1]
        # Remove return for submission
        if do_cv:
            return self.cnn_model.fit(X, y, epochs=self.epochs, batch_size=128, verbose=1,
                                      validation_split=0.2)
        else:
            return self.cnn_model.fit(X, y, epochs=self.epochs, batch_size=128, verbose=0)

    def predict(self, X):
        _, x = X
        return self.cnn_model.predict(X).ravel() + x[:, self.len_sequences-1, 1]
