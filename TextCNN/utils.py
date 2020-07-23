import os
import numpy as np
from collections import Counter
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


def read_file(file_name):
    contents = []
    labels = []
    with open(file_name, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            try:
                raw = line.strip().split("\t")
                content = raw[1].split(' ')
                if content:
                    contents.append(content)
                    labels.append(raw[0])
            except:
                pass
    return contents, labels


def read_single_file(file_name):
    contents = []
    label = file_name.split('/')[-1].split('.')[0]
    with open(file_name, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            try:
                content = line.strip().split(' ')
                if content:
                    contents.append(content)
            except:
                pass
    return contents, label


def read_files(directory):
    contents = []
    labels = []
    files = [f for f in os.listdir(directory) if f.endswith(".txt")]
    for file_name in files:
        content, label = read_single_file(os.path.join(directory, file_name))
        contents.extend(content)
        labels.extend([label] * len(content))
    return contents, labels


def build_vocab(train_dir, vocab_file, vocab_size=5000):
    data_train, _ = read_files(train_dir)

    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    words = ['<PAD>'] + list(words)
    open(vocab_file, mode='w', encoding='utf-8', errors='ignore').write('\n'.join(words) + '\n')


def read_vocab(vocab_file):
    with open(vocab_file, mode='r', encoding='utf-8', errors='ignore') as fp:
        words = [_.strip() for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def read_category():
    categories = ['car', 'entertainment', 'military', 'sports', 'technology']
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id


def encode_sentences(contents, words):
    return [encode_cate(x, words) for x in contents]


def encode_cate(content, words):
    return [(words[x] if x in words else 40000) for x in content]


def process_file(file_name, word_to_id, cat_to_id, max_length=600):
    contents, labels = read_file(file_name)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    x_pad = pad_sequences(data_id, max_length)
    y_pad = to_categorical(label_id, num_classes=len(cat_to_id))

    return x_pad, y_pad


def batch_iter(x, y, batch_size=64):
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
