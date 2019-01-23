import sys
import csv
import numpy as np
from sklearn.cluster import KMeans

image_npy = sys.argv[1]
test_csv = sys.argv[2]
pred_csv = sys.argv[3]

def output(y, filename):
    output = open(filename, 'w')
    output.write('ID,Ans\n')
    id = 0
    for row in test_data:
        id1 = int(row[1])
        id2 = int(row[2])
        if y[id1] == y[id2]:
            result = 1
        else:
            result = 0
        output.write('{},{}\n'.format(row[0], result))
        id+=1

if __name__ == '__main__':
    imgs = np.load(image_npy).astype(np.float64)
    text = open(test_csv, 'r')
    test = csv.reader(text, delimiter=',')

    counter = 0
    test_data = []
    for row in test:
        if counter == 0:
            counter += 1
            continue
        test_data.append(row)

    kmeans = np.load('kmeans.npy').item()
    output(kmeans.labels_, pred_csv)
    