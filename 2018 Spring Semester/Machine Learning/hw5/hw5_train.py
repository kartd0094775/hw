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
    train_label = f.readlines()
    f.close()
with open(sys.argv[2], 'r', encoding='utf-8') as f:
    train_nolabel = f.readlines()
    f.close()

def build_tokenizer():
    texts = []
    tokenizer = Tokenizer(num_words=vocab_size, filters='')
    for row in train_label:
        sentence = row.split(' +++$+++ ')[1]
        texts.append(sentence)
    for row in train_nolabel:
        texts.append(row)

    tokenizer.fit_on_texts(texts)
    return tokenizer
    
def to_sequence(tokenizer):
    train_label_x = []
    train_label_y = []
    train_nolabel_x = []

    for row in train_label:
        row = row.split(' +++$+++ ') 
        train_label_x.append(row[1])
        train_label_y.append(int(row[0]))
    for row in train_nolabel:
        train_nolabel_x.append(row)
        
    train_label_x = tokenizer.texts_to_sequences(train_label_x)
    train_label_x = np.array(pad_sequences(train_label_x, max_length, padding='post'))
    
    train_nolabel_x = tokenizer.texts_to_sequences(train_nolabel_x)
    train_nolabel_x = np.array(pad_sequences(train_nolabel_x, max_length, padding='post'))

    return train_label_x, train_label_y, train_nolabel_x

def validation(x, y):
    index = len(x) // 10
    train_x = x[index:]
    train_y = y[index:]
    validation_x = x[:index]
    validation_y = y[:index]
    return train_x, train_y, validation_x, validation_y

def build_wv_model():
    sentences = []
    for row in train_label:
        sentence = row.split(' +++$+++ ')[1]
        sentences.append(text_to_word_sequence(sentence, filters=''))  
    for row in train_nolabel:
        sentences.append(text_to_word_sequence(row, filters=''))
    return Word2Vec(sentences, size=embedding_dim, window=5, min_count=5)

def build_embedding_matrix(wv_model, tokenizer):
    embeddings_index = {}
    for k, v in wv_model.wv.vocab.items():
        embeddings_index[k] = wv_model.wv[k]
    embedding_matrix = np.zeros(shape=(vocab_size + 1, embedding_dim))
    for word, i in tokenizer.word_index.items():
        if i > 20000: break
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def semi_train(semi_data, model):
    # callback function
    earlystopping = EarlyStopping(monitor='val_acc', patience=3, verbose=1, mode='max')
    checkpoint = ModelCheckpoint('./hw5_semi_.w5', verbose=1, save_best_only=True, save_weights_only=True, monitor='val_acc', mode='max')
    
    # data segmentation
    data_slot = [semi_data[len(semi_data)//10 * k:len(semi_data)//10 * (k+1)] for k in range(10)]
    semi_x = train_x
    semi_y = train_y
    
    # first time semi-training
    pred = model.predict(data_slot[0], batch_size=1024, verbose=True)
    pred = np.squeeze(pred)
    label = state = np.greater(pred, 0.5).astype(np.int32)
    index = ((label > 0.8) | (label < 0.2))
    temp_x = np.concatenate((train_x, data_slot[0][index,:]))
    temp_y = np.concatenate((train_y, label[index]))

    model.fit(temp_x, temp_y, validation_data=(validation_x, validation_y), epochs=2, batch_size=batch_size, callbacks=[checkpoint, earlystopping])
    
    for i in range(9):
        pred = model.predict(np.concatenate((data_slot[i], data_slot[i+1])), batch_size=1024, verbose=True)
        pred = np.squeeze(pred)
        front_pred = pred[:len(pred)//2]
        label = np.greater(front_pred, 0.5).astype(np.int32)
        index = (label > 0.8) | (label < 0.2)
        index &= (label == state)
        
        semi_x = np.concatenate((semi_x, data_slot[i][index,:]))
        semi_y = np.concatenate((semi_y, label[index]))
        history = model.fit(semi_x, semi_y, validation_data=(validation_x, validation_y), epochs=2, batch_size=batch_size, callbacks=[checkpoint, earlystopping])
        
        back_pred = pred[len(pred)//2:]
        state = np.greater(back_pred, 0.5).astype(np.int32)

    pred = model.predict(data_slot[-1], batch_size=1024, verbose=True)
    pred = np.squeeze(pred)
    label = np.greater(pred, 0.5).astype(np.int32)
    index = (label > 0.8) | (label < 0.2)
    index &= (label == state)
    
    semi_x = np.concatenate((semi_x, data_slot[i][index,:]))
    semi_y = np.concatenate((semi_y, label[index]))
    history = model.fit(semi_x, semi_y, validation_data=(validation_x, validation_y), epochs=2, batch_size=batch_size, callbacks=[checkpoint, earlystopping])
    
    return model

max_length = 40
vocab_size = 20000
embedding_dim = 128
nb_epoch = 20
batch_size = 128

if __name__ == '__main__':

    tokenizer = build_tokenizer()
    pk.dump(tokenizer, open('tokenizer.pk', 'wb'))
    wv_model = build_wv_model()
    wv_model.save('wv_model')
    embedding_matrix = build_embedding_matrix(wv_model, tokenizer)
    train_label_x, train_label_y, train_nolabel_x = to_sequence(tokenizer)
    train_x, train_y, validation_x, validation_y = validation(train_label_x, train_label_y)
    np.save( 'embedding_matrix.npy', embedding_matrix)
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

    # training process
    checkpoint = ModelCheckpoint('./hw5_train_.w5', monitor='val_acc', save_weights_only=True, save_best_only=True, verbose=1)
    earlystopping = EarlyStopping(monitor='val_acc', patience=25, verbose=1, mode='max')
    model.fit(train_x, train_y, batch_size=batch_size, validation_data=(validation_x, validation_y), epochs=100, callbacks=[checkpoint, earlystopping])
    semi_train(train_nolabel_x, model)