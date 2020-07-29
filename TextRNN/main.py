import argparse
import random
from TextRNN.text_rnn import TextRNN
from util.metrics import evaluate
from util.news_data_util import *
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import logging
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--data_dir", type=str, default="../data/news_dataset", help="data file")
    parser.add_argument("--vocab_file", type=str, default="../data/vocab/vocab.txt", help="vocab_file")
    parser.add_argument("--vocab_size", type=int, default="40000", help="vocab_size")
    parser.add_argument("--max_features", type=int, default=40001, help="max_features")
    parser.add_argument("--max_len", type=int, default=100, help="max_len")
    parser.add_argument("--batch_size", type=int, default=64, help="batch_size")
    parser.add_argument("--embedding_size", type=int, default=50, help="embedding_size")
    parser.add_argument("--epochs", type=int, default=3, help="epochs")
    args = parser.parse_args()

    logger.info('加载数据构建词汇表...')
    if not os.path.exists(args.vocab_file):
        build_vocab(args.data_dir, args.vocab_file, args.vocab_size)

    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(args.vocab_file)

    logger.info('加载数据...')
    data, label = read_files(args.data_dir)
    data = list(zip(data, label))
    random.shuffle(data)

    train_data, test_data = train_test_split(data)

    data_train = encode_sentences([content[0] for content in train_data], word_to_id)
    label_train = to_categorical(encode_cate([content[1] for content in train_data], cat_to_id))
    data_test = encode_sentences([content[0] for content in test_data], word_to_id)
    label_test = to_categorical(encode_cate([content[1] for content in test_data], cat_to_id))

    data_train = sequence.pad_sequences(data_train, maxlen=args.max_len)
    data_test = sequence.pad_sequences(data_test, maxlen=args.max_len)

    model = TextRNN(args.max_len, args.max_features, args.embedding_size).build_model()
    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

    logger.info('开始训练...')
    callbacks = [
        ModelCheckpoint('./model.h5', verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=2, mode='max')
    ]

    history = model.fit(data_train, label_train,
                        batch_size=args.batch_size,
                        epochs=args.epochs,
                        callbacks=callbacks,
                        validation_data=(data_test, label_test))

    model.summary()
    label_pre = model.predict(data_test)
    pred_argmax = label_pre.argmax(-1)
    label_test = label_test.argmax(-1)
    print(evaluate(label_test, pred_argmax, categories))