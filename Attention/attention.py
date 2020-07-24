from tensorflow.keras import backend
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.layers import Layer


class Attention(Layer):
    def __init__(self, step_dim,
                 W_reg=None, b_reg=None,
                 W_const=None, b_const=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_reg = regularizers.get(W_reg)
        self.b_reg = regularizers.get(b_reg)

        self.W_const = constraints.get(W_const)
        self.b_const = constraints.get(b_const)

        self.bias = bias
        self.step_dim = step_dim
        self.features_size = 0

        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_reg,
                                 constraint=self.W_const)
        self.features_size = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_reg,
                                     constraint=self.b_const)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):

        e = backend.reshape(
            backend.dot(backend.reshape(x, (-1, self.features_size)), backend.reshape(self.W, (self.features_size, 1))),
            (-1, self.step_dim))
        if self.bias:
            e += self.b
        e = backend.tanh(e)

        a = backend.exp(e)
        if mask is not None:
            a *= backend.cast(mask, backend.floatx())
        a /= backend.cast(backend.sum(a, axis=1, keepdims=True) + backend.epsilon(), backend.floatx())
        a = backend.expand_dims(a)

        c = backend.sum(a * x, axis=1)
        return c

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_size
