import sys
import csv
import tensorflow as tf
import numpy as np
from keras.callbacks import *
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.preprocessing import image

m1 = load_model('private.h5')
m2 = load_model('public.h5')
m3 = load_model('hw3_model2_train_18_0.64774.h5')
test_csv = 'test.csv'
save_csv = 'result.csv'


text2 = open(test_csv, 'r')
test = csv.reader(text2, delimiter=',')

def preprocessing(test):
    test_x = []
    counter = 0
    for row in test:
        if counter == 0: 
            counter += 1
            continue
        temp = []
        for pixel in row[1].split():
            temp.append(int(pixel))
        temp = np.array(temp).reshape(48, 48, 1)
        test_x.append(temp)
        counter += 1
    test_x = np.array(test_x)    
    return test_x

def normalization(test_x):
    mean = np.load('mean.npy')
    std = np.load('std.npy')
    return np.divide(test_x - mean, std)

def output(prob, filename):
    output = open(filename, 'w')
    output.write('id,label\n')
    id = 0
    for i in range(len(prob[0])):
        tmp = [0, 0, 0, 0, 0, 0, 0]
        
        r1 = prob[0][i].argmax()
        r2 = prob[1][i].argmax()
        r3 = prob[2][i].argmax()
        tmp[r1] += 1
        tmp[r2] += 1
        tmp[r3] += 1
        result = tmp.index(max(tmp))
        output.write('{},{}\n'.format(id, result))
        id+=1

if __name__ == '__main__':
    test_x = preprocessing(test)
    test_x = normalization(test_x)
    r1 = m1.predict(test_x)
    r2 = m2.predict(test_x)
    r3 = m3.predict(test_x)
    output([r1, r2, r3], save_csv)   