from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Conv1D, Embedding, Concatenate, GlobalMaxPooling1D


class TextCNN(object):
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
        embedding = Embedding(self.max_features, self.embedding_dims, input_length=self.max_len)(input)
        convs = []
        for kernel_size in [3, 4, 5]:
            c = Conv1D(128, kernel_size, activation='relu')(embedding)
            c = GlobalMaxPooling1D()(c)
            convs.append(c)
        x = Concatenate()(convs)

        output = Dense(self.class_num, activation=self.activation)(x)
        model = Model(inputs=input, outputs=output)
        return model
