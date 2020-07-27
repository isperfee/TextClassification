from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Conv1D, Embedding, Concatenate, GlobalMaxPooling1D, GlobalMaxPool1D, Attention


class TextCNNAttention(object):
    def __init__(self, max_len, max_features, embedding_dims,
                 class_num=5,
                 activation='softmax'):
        self.max_len = max_len
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.activation = activation

    def build_model(self):
        input = Input((self.max_len,))
        query_embedding = Embedding(self.max_features, self.embedding_dims, input_length=self.max_len)(input)
        value_embedding = Embedding(self.max_features, self.embedding_dims, input_length=self.max_len)(input)
        convs = []
        for kernel_size in [3, 3, 3]:
            query_conv = Conv1D(128, kernel_size, activation='relu')(query_embedding)
            value_conv = Conv1D(128, kernel_size, activation='relu')(value_embedding)
        query_value_attention = Attention(self.max_len)([query_conv, value_conv])
        query_encoding = GlobalMaxPool1D()(query_conv)
        query_value_attention = GlobalMaxPool1D()(query_value_attention)
        concate = Concatenate(axis=-1)([query_encoding, query_value_attention])
        output = Dense(self.class_num, activation=self.activation)(concate)
        model = Model(inputs=input, outputs=output)
        return model
