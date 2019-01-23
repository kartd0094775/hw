import numpy as np
import matplotlib.pyplot as plt
import operator
import csv
import math
import sys

def test(model, input_data): 
    w = model
    result = []
    for i in range(len(input_data)):
        result.append(np.dot(w, input_data[i]))
    return result
def outputFile(result, input_file, filename):
    output = open(filename, 'w')
    output.write('id,value\n')
    for i in range(int(len(input_file)/18)):
        output_id = input_file[i * 18].split(',')[0]
        output_value = result[i]
        if output_value < 0:
            output_value = 0
        output.write(output_id +','+str(output_value) +'\n')
def generateTestData():
    test_x = []
    n_row = 0
    text = open(sys.argv[1], 'r', encoding='big5') 
    row = csv.reader(text , delimiter= ",")

    for r in row:
        if n_row %18 == 0:
            test_x.append([])
            for i in range(2,11):
                test_x[n_row//18].append(float(r[i]) )
        else :
            for i in range(2,11):
                if r[i] !="NR":
                    test_x[n_row//18].append(float(r[i]))
                else:
                    test_x[n_row//18].append(0)
        n_row = n_row+1
    text.close()
    test_x = np.array(test_x)

    # test_x = np.concatenate((test_x,test_x**2), axis=1)
    # 增加平方項

    test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)
    # 增加bias項  
    return test_x

if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    model = np.load('model_hw1.npy')
    with open(input_file, mode='r', encoding='utf8') as f:
        test_file = f.readlines()
        f.close
    test_data = generateTestData()
    output = test(model, test_data)
    outputFile(output, test_file, output_file)

