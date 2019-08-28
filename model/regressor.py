import time
import numpy as np
from sklearn.base import BaseEstimator
from keras.layers import Concatenate, Dropout, BatchNormalization, Activation, Dense, Input, \
    Flatten, Conv2D, Conv1D, MaxPooling2D, LSTM, Permute, RepeatVector, Multiply, Lambda, Add
from keras.models import Model
from keras.regularizers import l2
import keras.backend as K
from keras import initializers
from keras.engine import InputSpec
from keras.layers import Wrapper


class ConcreteDropout(Wrapper):
    """This wrapper allows to learn the dropout probability for any given input Dense layer.
    ```python
        # as the first layer in a model
        model = Sequential()
        model.add(ConcreteDropout(Dense(8), input_shape=(16)))
        # now model.output_shape == (None, 8)
        # subsequent layers: no need for input_shape
        model.add(ConcreteDropout(Dense(32)))
        # now model.output_shape == (None, 32)
    ```
    `ConcreteDropout` can be used with arbitrary layers which have 2D
    kernels, not just `Dense`. However, Conv2D layers require different
    weighing of the regulariser (use SpatialConcreteDropout instead).
    # Arguments
        layer: a layer instance.
        weight_regularizer:
            A positive number which satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$ (inverse observation noise),
            and N the number of instances in the dataset.
            Note that kernel_regularizer is not needed.
        dropout_regularizer:
            A positive number which satisfies
                $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and N the number of
            instances in the dataset.
            Note the relation between dropout_regularizer and weight_regularizer:
                $weight_regularizer / dropout_regularizer = l**2 / 2$
            with prior lengthscale l. Note also that the factor of two should be
            ignored for cross-entropy loss, and used only for the eculedian loss.
    """

    def __init__(self, layer, weight_regularizer=1e-6, dropout_regularizer=1e-5,
                 init_min=0.1, init_max=0.1, is_mc_dropout=True, **kwargs):
        assert 'kernel_regularizer' not in kwargs
        super(ConcreteDropout, self).__init__(layer, **kwargs)
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.is_mc_dropout = is_mc_dropout
        self.supports_masking = True
        self.p_logit = None
        self.p = None
        self.init_min = np.log(init_min) - np.log(1. - init_min)
        self.init_max = np.log(init_max) - np.log(1. - init_max)

    def build(self, input_shape=None):
        self.input_spec = InputSpec(shape=input_shape)
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super(ConcreteDropout, self).build()
        # this is very weird.. we must call super before we add new losses

        # initialise p
        self.p_logit = self.layer.add_weight(name='p_logit',
                                             shape=(1,),
                                             initializer=initializers.RandomUniform(self.init_min,
                                                                                    self.init_max),
                                             trainable=True)
        self.p = K.sigmoid(self.p_logit[0])

        # initialise regulariser / prior KL term
        assert len(input_shape) == 2, 'this wrapper only supports Dense layers'
        input_dim = np.prod(input_shape[-1])  # we drop only last dim
        weight = self.layer.kernel
        kernel_regularizer = self.weight_regularizer * K.sum(K.square(weight)) / (1. - self.p)
        dropout_regularizer = self.p * K.log(self.p)
        dropout_regularizer += (1. - self.p) * K.log(1. - self.p)
        dropout_regularizer *= self.dropout_regularizer * input_dim
        regularizer = K.sum(kernel_regularizer + dropout_regularizer)
        self.layer.add_loss(regularizer)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def concrete_dropout(self, x):
        '''
        Concrete dropout - used at training time (gradients can be propagated)
        :param x: input
        :return:  approx. dropped out input
        '''
        eps = K.cast_to_floatx(K.epsilon())
        temp = 0.1

        unif_noise = K.random_uniform(shape=K.shape(x))
        drop_prob = (
            K.log(self.p + eps)
            - K.log(1. - self.p + eps)
            + K.log(unif_noise + eps)
            - K.log(1. - unif_noise + eps)
        )
        drop_prob = K.sigmoid(drop_prob / temp)
        random_tensor = 1. - drop_prob

        retain_prob = 1. - self.p
        x *= random_tensor
        x /= retain_prob
        return x

    def call(self, inputs, training=None):
        if self.is_mc_dropout:
            return self.layer.call(self.concrete_dropout(inputs))
        else:
            def relaxed_dropped_inputs():
                return self.layer.call(self.concrete_dropout(inputs))
            return K.in_train_phase(relaxed_dropped_inputs,
                                    self.layer.call(inputs),
                                    training=training)


class Regressor(BaseEstimator):
    def __init__(self, num_scalar=12, num_const=7, epochs=200, len_sequences=10, dropout=0.3):
        self.epochs = epochs
        self.len_sequences = len_sequences
        len_lstm = 4

        l2_weight = 1
        l2_lstm = 10
        l2_conv = 10

        img_in = Input(shape=(11, 11, 7))
        scalar_in = Input(shape=(len_sequences, num_scalar))
        const_in = Input(shape=(num_const,))

        # model_img = BatchNormalization()(img_in)
        model_img = img_in
        model_img = Conv2D(32, (5, 5), padding="same", kernel_regularizer=l2(l2_conv))(model_img)
        model_img = Activation("selu")(model_img)

        img_shortcut = model_img
        # model_img = BatchNormalization()(model_img)
        model_img = Conv2D(32, (5, 5), padding="same", kernel_regularizer=l2(l2_conv))(model_img)
        model_img = Activation("selu")(model_img)

        # model_img = BatchNormalization()(model_img)
        model_img = Conv2D(32, (5, 5), padding="same", kernel_regularizer=l2(l2_conv))(model_img)
        model_img = Add()([model_img, img_shortcut])
        model_img = Activation("selu")(model_img)
        model_img = MaxPooling2D()(model_img)

        # model_img = BatchNormalization()(model_img)
        model_img = Conv2D(64, (3, 3), padding="same", kernel_regularizer=l2(l2_conv))(model_img)
        model_img = Activation("selu")(model_img)

        img_shortcut = model_img
        # model_img = BatchNormalization()(model_img)
        model_img = Conv2D(64, (3, 3), padding="same", kernel_regularizer=l2(l2_conv))(model_img)
        model_img = Activation("selu")(model_img)

        # model_img = BatchNormalization()(model_img)
        model_img = Conv2D(64, (3, 3), padding="same", kernel_regularizer=l2(l2_conv))(model_img)
        model_img = Add()([model_img, img_shortcut])
        model_img = Activation("selu")(model_img)
        model_img = MaxPooling2D()(model_img)

        # model_img = BatchNormalization()(model_img)
        model_img = Conv2D(128, (3, 3), padding="same", kernel_regularizer=l2(l2_conv))(model_img)
        model_img = Activation("selu")(model_img)

        img_shortcut = model_img
        # model_img = BatchNormalization()(model_img)
        model_img = Conv2D(128, (3, 3), padding="same", kernel_regularizer=l2(l2_conv))(model_img)
        model_img = Activation("selu")(model_img)

        # model_img = BatchNormalization()(model_img)
        model_img = Conv2D(128, (3, 3), padding="same", kernel_regularizer=l2(l2_conv))(model_img)
        model_img = Add()([model_img, img_shortcut])
        model_img = Activation("selu")(model_img)
        model_img = MaxPooling2D()(model_img)

        model_img = Flatten()(model_img)

        model_img = ConcreteDropout(Dense(128, kernel_regularizer=l2(l2_weight)))(model_img)
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
        # model_scalar_2 = MaxPooling1D()(model_scalar_2)
        model_scalar_2 = Flatten()(model_scalar_2)
        model_scalar_2 = Dense(16, kernel_regularizer=l2(l2_weight))(model_scalar_2)
        model_scalar_2 = Dropout(0.3)(model_scalar_2)
        model_scalar_2 = Activation("tanh")(model_scalar_2)

        # model_const = BatchNormalization()(const_in)
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
            history = self.cnn_model.fit(X, y, epochs=self.epochs, batch_size=128, verbose=1,
                                         validation_split=0.2)
        else:
            history = self.cnn_model.fit(X, y, epochs=self.epochs, batch_size=128, verbose=1)
        print("Training done in {:.0f}s".format(time.time()-t))
        return history

    def predict(self, X):
        _, x, _ = X
        nb_mean = 200
        pred = self.cnn_model.predict(X).ravel() + x[:, self.len_sequences-1, 1]
        for i in range(nb_mean-1):
            pred += self.cnn_model.predict(X).ravel() + x[:, self.len_sequences-1, 1]
        return np.around(pred/nb_mean, decimals=0)
