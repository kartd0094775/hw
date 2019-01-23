from PIL import Image
import numpy as np
import os, csv, sys, pickle

# usage:
# python3 find_eqaul.py train.csv <predict.csv> <output.csv>

#reader = csv.reader(open('train.csv', 'r'))
reader = csv.reader(open(sys.argv[1], 'r'))
trainfile_id_dict = {}
for row in reader:
    if row[0] == 'Image':
        continue
    trainfile_id_dict[row[0]] = row[1]

d = np.load('data/test_equals_train_dict.npy').item()

test_id_dict = {}
for testfile, trainfile in d.items():
    test_id_dict[testfile] = trainfile_id_dict[trainfile[0]]


reader = csv.reader(open(sys.argv[2], 'r')) #source predict csv
writer = csv.writer(open(sys.argv[3], 'w', newline='')) #output csv
#reader = csv.reader(open('sample_submission.csv', 'r')) #source predict csv
#writer = csv.writer(open('test.csv', 'w', newline='')) #output csv
cnt=0
for row in reader:
    if row[0] in d.keys():
        
        print('##found')
        print(row)
        print(trainfile_id_dict[d[row[0]][0]])
        tmp = row[1].split()
        if tmp[0] != trainfile_id_dict[d[row[0]][0]]:
            cnt+=1
            print('num of replaced:',cnt)
        row[1] = trainfile_id_dict[d[row[0]][0]] + ' '
        for i in range(4):
            row[1] += tmp[i] + ' '
    writer.writerow(row)