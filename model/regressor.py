from keras.layers import Concatenate, Dropout, BatchNormalization, Activation, Dense, Input, \
    Flatten, Conv2D, Conv1D, MaxPooling2D, LSTM
from keras.models import Model
from keras.regularizers import l2
from sklearn.base import BaseEstimator


class Regressor(BaseEstimator):
    def __init__(self, epochs=300, len_sequences=5):
        self.epochs = epochs
        self.len_sequences = len_sequences
        l2_weight = 5
        img_in = Input(shape=(11, 11, 7))
        scalar_in = Input(shape=(len_sequences, 9))
        const_in = Input(shape=(3,))

        model_img = BatchNormalization()(img_in)
        model_img = Conv2D(512, (5, 5), padding="same")(model_img)
        model_img = Activation("relu")(model_img)
        model_img = MaxPooling2D()(model_img)

        model_img = BatchNormalization()(model_img)
        model_img = Conv2D(256, (5, 5), padding="same")(model_img)
        model_img = Activation("relu")(model_img)
        model_img = MaxPooling2D()(model_img)

        model_img = BatchNormalization()(model_img)
        model_img = Conv2D(128, (3, 3), padding="same")(model_img)
        model_img = Activation("relu")(model_img)
        model_img = Flatten()(model_img)

        model_img = Dense(128, kernel_regularizer=l2(l2_weight))(model_img)
        model_img = Activation("tanh")(model_img)

        model_scalar = LSTM(32, activation='tanh', kernel_regularizer=l2(l2_weight))(scalar_in)
        # model_scalar = Conv1D(64, 3, padding="same")(scalar_in)
        # model_scalar = Flatten()(model_scalar)
        model_scalar = Dense(128)(model_scalar)

        model_const = Dense(128, kernel_regularizer=l2(l2_weight))(const_in)

        model = Concatenate()([model_img, model_scalar, model_const])
        model = BatchNormalization()(model)

        model = Dense(64, kernel_regularizer=l2(l2_weight))(model)
        model = Activation("tanh")(model)

        model = Dense(32, kernel_regularizer=l2(l2_weight))(model)
        model = Dropout(0.5)(model)
        model = Activation("tanh")(model)

        model = Dense(1)(model)

        self.cnn_model = Model([img_in, scalar_in, const_in], model)
        self.cnn_model.compile(loss="mse", optimizer="adam")

        print(self.cnn_model.summary())
        return

    def fit(self, X, y, do_cv=False):
        _, x, _ = X
        y = y - x[:, self.len_sequences-1, 1]
        # Remove return for submission
        if do_cv:
            return self.cnn_model.fit(X, y, epochs=self.epochs, batch_size=128, verbose=1,
                                      validation_split=0.2)
        else:
            return self.cnn_model.fit(X, y, epochs=self.epochs, batch_size=128, verbose=1)

    def predict(self, X):
        _, x, _ = X
        return self.cnn_model.predict(X).ravel() + x[:, self.len_sequences-1, 1]
