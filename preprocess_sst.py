import numpy as np
import pickle
import os
from bpe_encoder import TextEncoder

def read_sentence(path):

    sents = []
    labels = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            labels.append(int(line[0]))
            sents.append(line[2:].replace('\n', ''))

    return sents, labels

encoder_path = './pretrain/encoder_bpe_40000.json'
bpe_path = './pretrain/vocab_40000.bpe'

root = './data'


train_sent_path = os.path.join(root, 'stsa.binary.train')
val_sent_path = os.path.join(root, 'stsa.binary.dev')
test_sent_path = os.path.join(root, 'stsa.binary.test')

trainX1, trainY = read_sentence(train_sent_path)
devX1, devY = read_sentence(val_sent_path)
testX1, testY = read_sentence(test_sent_path)


print(len(trainY))
print(len(devY))
print(len(testY))

sentences = trainX1 + devX1 + testX1
labels = trainY + devY + testY

# bpe encoder
text_encoder = TextEncoder(encoder_path, bpe_path)
encoder = text_encoder.encoder
bpe_sentences = text_encoder.encode(sentences)

n_vocab = len(encoder)
max_word = 100
sst_array = np.zeros([len(bpe_sentences), max_word, 2], dtype=np.int32)
sst_mask = np.zeros([len(bpe_sentences), max_word], dtype=np.int32)

encoder['_start_'] = len(encoder)
encoder['_delimiter_'] = len(encoder)
encoder['_classify_'] = len(encoder)

for i, line, in enumerate(bpe_sentences):

    x = [encoder['_start_']] + line + [encoder['_classify_']]
    sl = len(x)
    sst_array[i, :sl, 0] = x
    sst_mask[i, :sl] = 1
sst_array[:, :, 1] = np.arange(n_vocab + 3, n_vocab + 3 + max_word)


sst_dump = open('./data/sst_encoded.pkl', 'wb')
pickle.dump(sst_array, sst_dump)
sst_dump.close()

sst_mask_dump = open('./data/sst_mask.pkl', 'wb')
pickle.dump(sst_mask, sst_mask_dump)
sst_mask_dump.close()

lb_dump = open('./data/sst_label.pkl', 'wb')
pickle.dump(np.array(labels, dtype=np.int32), lb_dump)
lb_dump.close()