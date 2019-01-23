import numpy as np
import csv
import sys

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
    return result
def sigmoid(x):
    return np.divide(1, (1 + np.exp(-x)))
def predict(w, b, test, filename):
    z = np.dot(test, w) + b
    prob = sigmoid(z)
    output = open(filename, 'w')
    output.write('id,label\n')
    id = 1
    for i in prob:
        result = 0
        if i >= 0.5:
            result = 1
        output.write('{},{}\n'.format(id, result))
        id+=1
if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    w = np.load('generative_w.npy')
    b = np.load('generative_b.npy')
    text = open(input_file, 'r')
    test_file = csv.reader(text, delimiter=',')
    test = preprocessing(test_file)
    predict(w, b, test, output_file)

