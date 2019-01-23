import sys
import csv
import tensorflow as tf
import numpy as np
from PIL import Image
from keras.callbacks import *
from keras.models import Sequential, load_model
from keras.layers import ZeroPadding2D, BatchNormalization, Conv2D, PReLU, MaxPooling2D, Flatten
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.preprocessing import image
from keras.callbacks import *

test_csv = sys.argv[1]
save_csv = sys.argv[2]
model_name = sys.argv[3]

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
    for row in prob:
        result = row.argmax()
        output.write('{},{}\n'.format(id, result))
        id+=1

if __name__ == '__main__':
    test_x = preprocessing(test)
    test_x = normalization(test_x)
    model = load_model(model_name)
    test_y = model.predict(test_x)
    output(test_y, save_csv)   