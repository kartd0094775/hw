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
    tokenizer = pk.load(open('public_token.pk', 'rb'))
    embedding_matrix = np.load('embedding_matrix.npy')
    test_x = to_sequence(tokenizer)
    # RNN model
    inputs = Input(shape=(max_length,))
    embedding_inputs = Embedding(vocab_size+1, embedding_dim, weights=[embedding_matrix], input_length=max_length, trainable=False)(inputs)
    RNN_cell = Bidirectional(LSTM(512, return_sequences=True, dropout=0.3))
    RNN_outputs = RNN_cell(embedding_inputs)
    RNN_cell = Bidirectional(LSTM(512, return_sequences=True, dropout=0.3))
    RNN_outputs = RNN_cell(RNN_outputs)
    RNN_cell = Bidirectional(LSTM(512, return_sequences=False, dropout=0.3))
    RNN_outputs = RNN_cell(RNN_outputs)
    outputs = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.1))(RNN_outputs)
    outputs = Dropout(0.3)(outputs)
    outputs = Dense(1, activation='sigmoid')(outputs)
    model = Model(inputs=inputs, outputs=outputs)
    adam = Adam()
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy',])
    model.summary()
    model.load_weights('public.w5')
    pred = model.predict(test_x, batch_size=1024, verbose=True)
    pred = np.squeeze(pred)
    test_y = np.greater(pred, 0.5).astype('int32')
    output = [(i, test_y[i]) for i in range(len(test_y))]
    dw = pd.DataFrame(output, columns = ["id", "label"])
    dw.to_csv(csv_path, index=False)
