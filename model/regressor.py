import time
from sklearn.base import BaseEstimator
from keras.layers import Concatenate, Dropout, BatchNormalization, \
     Activation, Dense, Input, Flatten, Conv2D, Conv1D, MaxPooling2D, \
     LSTM, Permute, RepeatVector, Multiply, Lambda, Add
from keras.models import Model
from keras.regularizers import l2
import keras.backend as K
from keras.layers import LeakyReLU


class Regressor(BaseEstimator):
    def __init__(self, num_scalar=12, num_const=7, epochs=200, len_sequences=10):
        self.epochs = epochs
        self.len_sequences = len_sequences
        len_lstm = 4

        l2_weight = 1
        l2_lstm = 10
        l2_conv = 10

        img_in = Input(shape=(11, 11, 7))
        scalar_in = Input(shape=(len_sequences, num_scalar))
        const_in = Input(shape=(num_const,))

        model_img = img_in
        model_img = Conv2D(32, (5, 5), padding="same", kernel_regularizer=l2(l2_conv))(model_img)
        model_img = LeakyReLU(alpha=0.5)(model_img)

        img_shortcut = model_img
        model_img = Conv2D(32, (5, 5), padding="same", kernel_regularizer=l2(l2_conv))(model_img)
        model_img = LeakyReLU(alpha=0.5)(model_img)

        model_img = Conv2D(32, (5, 5), padding="same", kernel_regularizer=l2(l2_conv))(model_img)
        model_img = Add()([model_img, img_shortcut])
        model_img = LeakyReLU(alpha=0.5)(model_img)
        model_img = MaxPooling2D()(model_img)

        model_img = Conv2D(64, (3, 3), padding="same", kernel_regularizer=l2(l2_conv))(model_img)
        model_img = LeakyReLU(alpha=0.5)(model_img)

        img_shortcut = model_img
        model_img = Conv2D(64, (3, 3), padding="same", kernel_regularizer=l2(l2_conv))(model_img)
        model_img = LeakyReLU(alpha=0.5)(model_img)

        model_img = Conv2D(64, (3, 3), padding="same", kernel_regularizer=l2(l2_conv))(model_img)
        model_img = Add()([model_img, img_shortcut])
        model_img = LeakyReLU(alpha=0.5)(model_img)
        model_img = MaxPooling2D()(model_img)

        model_img = Conv2D(128, (3, 3), padding="same", kernel_regularizer=l2(l2_conv))(model_img)
        model_img = LeakyReLU(alpha=0.5)(model_img)

        img_shortcut = model_img
        model_img = Conv2D(128, (3, 3), padding="same", kernel_regularizer=l2(l2_conv))(model_img)
        model_img = LeakyReLU(alpha=0.5)(model_img)

        model_img = Conv2D(128, (3, 3), padding="same", kernel_regularizer=l2(l2_conv))(model_img)
        model_img = Add()([model_img, img_shortcut])
        model_img = LeakyReLU(alpha=0.5)(model_img)
        model_img = MaxPooling2D()(model_img)

        model_img = Flatten()(model_img)

        model_img = Dense(128, kernel_regularizer=l2(l2_weight))(model_img)
        model_img = Dropout(0.15)(model_img)
        model_img = Activation("tanh")(model_img)

        model_scalar = scalar_in
        activations = LSTM(len_lstm, activation='tanh', kernel_regularizer=l2(l2_lstm),
                           return_sequences=True)(model_scalar)
        attention = Dense(1, activation='tanh')(activations)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(len_lstm)(attention)
        attention = Permute([2, 1])(attention)

        sent_representation = Multiply()([activations, attention])
        sent_representation = Lambda(lambda xin: K.sum(xin, axis=-2),
                                     output_shape=(len_lstm,))(sent_representation)
        model_scalar = Dense(128)(sent_representation)
        model_scalar = Dense(64)(sent_representation)
        model_scalar = Activation("tanh")(model_scalar)

        model_scalar_2 = Conv1D(64, 3, padding="same")(scalar_in)
        model_scalar_2 = Activation("selu")(model_scalar_2)
        model_scalar_2 = Flatten()(model_scalar_2)
        model_scalar_2 = Dense(16, kernel_regularizer=l2(l2_weight))(model_scalar_2)
        model_scalar_2 = Dropout(0.3)(model_scalar_2)
        model_scalar_2 = Activation("tanh")(model_scalar_2)

        model_const = const_in
        model_const = Dense(64, kernel_regularizer=l2(l2_weight))(model_const)
        model_const = Activation("tanh")(model_const)

        model = Concatenate()([model_img, model_scalar, model_scalar_2, model_const])
        model = BatchNormalization()(model)

        model = Dense(64, kernel_regularizer=l2(l2_weight))(model)
        model = Activation("tanh")(model)
        model = Dense(32, kernel_regularizer=l2(l2_weight))(model)
        model = Dropout(0.3)(model)
        model = Activation("tanh")(model)

        model = Dense(16, kernel_regularizer=l2(l2_weight))(model)
        model = Activation("tanh")(model)

        model = Dense(1)(model)

        self.cnn_model = Model([img_in, scalar_in, const_in], model)
        self.cnn_model.compile(loss="mse", optimizer="adam")

        print(self.cnn_model.summary())
        return

    def fit(self, X, y, do_cv=False):
        t = time.time()
        _, x, _ = X
        y = y - x[:, self.len_sequences-1, 1]
        if do_cv:
            history = self.cnn_model.fit(X, y, epochs=self.epochs,
                                         batch_size=128, verbose=1,
                                         validation_split=0.2)
        else:
            history = self.cnn_model.fit(X, y, epochs=self.epochs,
                                         batch_size=128, verbose=1)
        print("Training done in {:.0f}s".format(time.time()-t))
        return history

    def predict(self, X):
        _, x, _ = X
        pred = self.cnn_model.predict(X).ravel() + x[:, self.len_sequences-1, 1]
        return pred
