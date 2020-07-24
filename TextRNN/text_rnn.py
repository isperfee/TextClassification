from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Embedding, Dense, LSTM


class TextRNN(object):
    def __init__(self, max_len, max_features, embedding_size,
                 class_num=5,
                 activation='softmax'):
        self.max_len = max_len
        self.max_features = max_features
        self.embedding_size = embedding_size
        self.class_num = class_num
        self.activation = activation

    def build_model(self):
        input = Input((self.max_len,))

        embedding = Embedding(self.max_features, self.embedding_size, input_length=self.max_len)(input)
        x = LSTM(128)(embedding)

        output = Dense(self.class_num, activation=self.activation)(x)
        model = Model(inputs=input, outputs=output)
        return model