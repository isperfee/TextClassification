from tensorflow.keras import Input, Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Embedding, Dense, SimpleRNN, Lambda, Concatenate, Conv1D, GlobalMaxPooling1D


class TextRCNN(object):
    def __init__(self, max_len, max_features, embedding_size,
                 class_num=5,
                 activation='softmax'):
        self.max_len = max_len
        self.max_features = max_features
        self.embedding_size = embedding_size
        self.class_num = class_num
        self.activation = activation

    def build_model(self):
        input_current = Input((self.max_len,))
        input_left = Input((self.max_len,))
        input_right = Input((self.max_len,))

        embedder = Embedding(self.max_features, self.embedding_size, input_length=self.max_len)
        embedding_current = embedder(input_current)
        embedding_left = embedder(input_left)
        embedding_right = embedder(input_right)

        x_left = SimpleRNN(128, return_sequences=True)(embedding_left)
        x_right = SimpleRNN(128, return_sequences=True, go_backwards=True)(embedding_right)
        x_right = Lambda(lambda x: K.reverse(x, axes=1))(x_right)
        x = Concatenate(axis=2)([x_left, embedding_current, x_right])

        x = Conv1D(64, kernel_size=1, activation='tanh')(x)
        x = GlobalMaxPooling1D()(x)

        output = Dense(self.class_num, activation=self.activation)(x)
        model = Model(inputs=[input_current, input_left, input_right], outputs=output)
        return model
