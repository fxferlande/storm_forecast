import time
from keras.layers import Concatenate, Dropout, BatchNormalization, Activation, Dense, Input, \
    Flatten, Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, LSTM
from keras.models import Model
from keras.regularizers import l2
# from keras.optimizers import Adam
# from keras.callbacks import LearningRateScheduler
from sklearn.base import BaseEstimator


class Regressor(BaseEstimator):
    def __init__(self, epochs=300, len_sequences=5, reg=4, dropout=0.3):
        self.epochs = epochs
        self.len_sequences = len_sequences
        l2_weight = reg
        img_in = Input(shape=(11, 11, 7))
        img_in_2 = Input(shape=(11, 11, 7))
        scalar_in = Input(shape=(len_sequences, 13))
        scalar_in_2 = Input(shape=(len_sequences, 13))
        const_in = Input(shape=(3,))

        model_img = BatchNormalization()(img_in)
        model_img = Conv2D(128, (5, 5), padding="same")(model_img)
        model_img = Activation("relu")(model_img)
        model_img = MaxPooling2D()(model_img)

        model_img = BatchNormalization()(model_img)
        model_img = Conv2D(128, (5, 5), padding="same")(model_img)
        model_img = Activation("relu")(model_img)
        model_img = MaxPooling2D()(model_img)

        model_img = BatchNormalization()(model_img)
        model_img = Conv2D(64, (3, 3), padding="same")(model_img)
        model_img = Activation("relu")(model_img)
        model_img = Flatten()(model_img)

        model_img = Dense(64, kernel_regularizer=l2(l2_weight))(model_img)
        model_img = Activation("tanh")(model_img)

        model_img_2 = BatchNormalization()(img_in_2)
        model_img_2 = Conv2D(128, (7, 7), padding="same")(model_img_2)
        model_img_2 = Activation("relu")(model_img_2)
        model_img_2 = MaxPooling2D()(model_img_2)
        model_img_2 = Flatten()(model_img_2)
        model_img_2 = Dense(64, kernel_regularizer=l2(l2_weight))(model_img_2)
        model_img_2 = Activation("tanh")(model_img_2)

        model_scalar = LSTM(64, activation='tanh', kernel_regularizer=l2(l2_weight))(scalar_in)
        model_scalar = Dense(128)(model_scalar)
        model_scalar = Dense(64)(model_scalar)
        model_scalar = Activation("tanh")(model_scalar)

        model_scalar_2 = BatchNormalization()(scalar_in_2)
        model_scalar_2 = Conv1D(128, 3, padding="same")(model_scalar_2)
        model_scalar_2 = Activation("relu")(model_scalar_2)
        model_scalar_2 = MaxPooling1D()(model_scalar_2)
        model_scalar_2 = Flatten()(model_scalar_2)
        model_scalar_2 = Dense(64)(model_scalar_2)
        model_scalar_2 = Activation("tanh")(model_scalar_2)

        model_const = Dense(64, kernel_regularizer=l2(l2_weight))(const_in)
        model_const = Activation("tanh")(model_const)

        model = Concatenate()([model_img, model_img_2, model_scalar, model_scalar_2, model_const])
        model = BatchNormalization()(model)

        model = Dense(64, kernel_regularizer=l2(l2_weight))(model)
        model = Dense(32, kernel_regularizer=l2(l2_weight))(model)
        model = Activation("tanh")(model)

        model = Dense(16, kernel_regularizer=l2(l2_weight))(model)
        model = Dropout(dropout)(model)
        model = Activation("tanh")(model)

        model = Dense(1)(model)

        # opt = Adam(lr=0.01, decay=0.05, clipvalue=0.001)
        self.cnn_model = Model([img_in, img_in_2, scalar_in, scalar_in_2, const_in], model)
        self.cnn_model.compile(loss="mse", optimizer="adam")

        print(self.cnn_model.summary())
        return

    # def learning_rate(self, epoch):
    #     init_curve = 0.1
    #     decay = 0.01
    #     limit = 0.0001
    #     if epoch < 10:
    #         return 1.0
    #     elif epoch < 100:
    #         return 0.1
    #     else:
    #         return float(max(init_curve*(1-decay)**(epoch-100), limit))

    def fit(self, X, y, do_cv=False):
        t = time.time()
        _, _, x, _, _ = X
        y = y - x[:, self.len_sequences-1, 1]
        # Remove return for submission
        # callback_lr = LearningRateScheduler(self.learning_rate, verbose=0)
        if do_cv:
            history = self.cnn_model.fit(X, y, epochs=self.epochs, batch_size=128, verbose=1,
                                         validation_split=0.2)  # , callbacks=[callback_lr])
        else:
            history = self.cnn_model.fit(X, y, epochs=self.epochs, batch_size=128, verbose=1)  # ,
            # callbacks=[callback_lr])
        print("Training done in {}".format(time.time()-t))
        return history

    def predict(self, X):
        _, _, x, _, _ = X
        return self.cnn_model.predict(X).ravel() + x[:, self.len_sequences-1, 1]
