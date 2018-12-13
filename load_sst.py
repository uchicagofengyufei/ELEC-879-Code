import numpy as np
import pickle
import os
from time import time
from bpe_encoder import TextEncoder

sst_pkl = './data/sst_encoded.pkl'
sst_label_pkl = './data/sst_label.pkl'
sst_mask_pkl = './data/sst_mask.pkl'
encoder_path = './pretrain/encoder_bpe_40000.json'
bpe_path = './pretrain/vocab_40000.bpe'

def array(x, dtype=np.int32):
    return np.array(x, dtype=dtype)


def load_pkl(file):
    # load pickle file
    f = open(file, 'rb')
    data = pickle.load(f)
    f.close()

    return data


class DataLoader:

    def __init__(self, max_word=100):
        # read label: list of tuple (date, 0/1)
        # news: list of np tuple (date, news array with encoding)
        # news array: (n_news, 30(max_word), 2)
        # n_lag1, n_lag2: scope of mid turn and long turn range
        # max_news maximun number of news

        self.sst = load_pkl(sst_pkl)
        self.mask = load_pkl(sst_mask_pkl)
        self.label = load_pkl(sst_label_pkl)

        # bpe encoder
        self.text_encoder = TextEncoder(encoder_path, bpe_path)
        self.encoder = self.text_encoder.encoder
        self.n_vocab = len(self.encoder)

        self.n_special = 3
        self.max_word = max_word

        self.pos = 0
        self.train_index = np.arange(0, 6920)
        self.val_index = np.arange(6920, 6920+872)
        self.test_index = np.arange(6920+872, 6920+872+1821)

        self.build_extra_embedding()

    def build_extra_embedding(self):
        self.encoder['_start_'] = len(self.encoder)
        self.encoder['_delimiter_'] = len(self.encoder)
        self.encoder['_classify_'] = len(self.encoder)

    def iter_reset(self, shuffle=True):
        self.pos = 0
        if shuffle:
            np.random.shuffle(self.train_index)

    def sampled_batch(self, batch_size, phase='train'):

        # batch iterator, shuffle if train
        if phase == 'train':
            n = len(self.train_index)
            self.iter_reset(shuffle=True)
            index = self.train_index
        elif phase == 'validation':
            n = len(self.val_index)
            self.iter_reset(shuffle=False)
            index = self.val_index
        else:
            n = len(self.test_index)
            self.iter_reset(shuffle=False)
            index = self.test_index



        while self.pos < n:

            X_batch = []
            y_batch = []
            M_batch = []

            for i in range(batch_size):
                X_batch.append(self.sst[index[self.pos]])
                y_batch.append(self.label[index[self.pos]])
                M_batch.append(self.mask[index[self.pos]])

                self.pos += 1
                if self.pos >= n:
                    break
            # all return news array are: [batch_size, max_news, max_words, 2]
            # all masks are              [batch_size, max_news, max_word]

            yield array(X_batch), array(y_batch), array(M_batch)


    def get_data(self, phase='train'):
        if phase == 'train':
            return self.sst[self.train_index], self.label[self.train_index], self.mask[self.train_index]
        elif phase == 'validation':
            return self.sst[self.val_index], self.label[self.val_index], self.mask[self.val_index]
        elif phase == 'test':
            return self.sst[self.test_index], self.label[self.test_index], self.mask[self.test_index]


if __name__ == '__main__':

    # text_encoder = TextEncoder(encoder_path, bpe_path)
    # encoder = text_encoder.encoder
    # for key,val in encoder.items():
    #    if key == '\251' or key == '\251</w>':
    #        print(key, val)
    # exit(0)
    # Testing
    iterator = DataLoader()
    # for sample in iterator.sampled_batch(1, phase='test'):
    #        print(sample[0], sample[1], sample[2])
    #        print(sample[3].shape)
    #        print(sample[4].shape)
    #        print(sample[5].shape)
    #        print(sample[6].shape)
    #        print(sample[7].shape)
    #        print('\n')

    decoder = iterator.text_encoder.decoder
    for x, y, m in iterator.sampled_batch(batch_size=1):
        print(x.shape)
        print(y.shape)
        print(m.shape)