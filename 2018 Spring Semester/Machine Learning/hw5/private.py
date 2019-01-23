import csv
import sys
import os
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.models import load_model, Model
from keras.layers import Input, Embedding, LSTM, Dropout, Dense, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import regularizers
from gensim.models import Word2Vec
import tensorflow as tf
import numpy as np
import pandas as pd
import _pickle as pk

with open(sys.argv[1], 'r', encoding='utf-8') as f:
    f.readline()
    test = f.readlines()
    f.close()

def to_sequence(tokenizer):
    test_x = []
    for row in test:
        index = row.find(',')
        test_x.append(row[index+1:])
        
    test_x = tokenizer.texts_to_sequences(test_x)
    test_x = np.array(pad_sequences(test_x, max_length, padding='post'))
    return test_x

max_length = 40
vocab_size = 20000
embedding_dim = 128
nb_epoch = 20
batch_size = 128

csv_path = sys.argv[2]

if __name__ == '__main__':
    tokenizer = pk.load(open('private_token.pk', 'rb'))
    test_x = to_sequence(tokenizer)
    model = load_model('private.h5')
    pred = model.predict(test_x, batch_size=1024, verbose=True)
    pred = np.squeeze(pred)
    test_y = np.greater(pred, 0.5).astype('int32')
    output = [(i, test_y[i]) for i in range(len(test_y))]
    dw = pd.DataFrame(output, columns = ["id", "label"])
    dw.to_csv(csv_path, index=False)
