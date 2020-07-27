from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Embedding, Dense, Bidirectional, LSTM, TimeDistributed
from Attention.attention import Attention


class HierarchicalAttentionNetwork(object):
    def __init__(self, max_len_sentence, max_len_word, max_features, embedding_size,
                 class_num=5,
                 activation='softmax'):
        self.max_len_sentence = max_len_sentence
        self.max_len_word = max_len_word
        self.max_features = max_features
        self.embedding_size = embedding_size
        self.class_num = class_num
        self.activation = activation

    def build_model(self):
        input_word = Input(shape=(self.max_len_word,))
        x_word = Embedding(self.max_features, self.embedding_size, input_length=self.max_len_word)(input_word)
        x_word = Bidirectional(LSTM(128, return_sequences=True))(x_word)
        x_word = Attention(self.max_len_word)(x_word)
        model_word = Model(input_word, x_word)

        input = Input(shape=(self.max_len_sentence, self.max_len_word))
        x_sentence = TimeDistributed(model_word)(input)
        x_sentence = Bidirectional(LSTM(128, return_sequences=True))(x_sentence)
        x_sentence = Attention(self.max_len_sentence)(x_sentence)

        output = Dense(self.class_num, activation=self.activation)(x_sentence)
        model = Model(inputs=input, outputs=output)
        return model
