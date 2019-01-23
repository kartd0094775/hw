import numpy as np
import csv
import sys
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.models import load_model


def preprocessing(test_x):
    result_test_x = []
    n_row = 0
    for row in test_x:
        temp = []
        if (n_row == 0):
            n_row += 1
            continue
        for i in range(len(row)):
            if i == 10: continue
            f = row[i]
            temp.append(float(f))
        result_test_x.append(temp)
        n_row += 1
    scaled = feature_scaling(np.array(result_test_x))
    return  scaled
def feature_scaling(test):
    mean = np.load('mean.npy')
    std = np.load('std.npy')
    result =  np.nan_to_num(np.divide((test - mean), std))
    result = np.concatenate((np.ones((result.shape[0], 1)), result), axis=1)
    return result
def output(prob, filename):
    output = open(filename, 'w')
    output.write('id,label\n')
    id = 1
    for row in prob:
        result = 0
        if row[0] <= row[1]:
            result = 1
        output.write('{},{}\n'.format(id, result))
        id+=1

if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    text = open(input_file, 'r')
    test_file = csv.reader(text, delimiter=',')
    test = preprocessing(test_file)
    model = load_model('best.h5')
    result = model.predict(test)
    output(result, output_file)
