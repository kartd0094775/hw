import sys, os, math, time
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

N_voter = 1; voters = []; N_test = -1; test_list = []
start = 0
# 5,6,7,10,11,12,14 -> 7 voters
must = ['']
for i in range(start, N_voter+start):
	# filename = 'predict'+str(i)+'.csv'
	filename = 'predict'+str(must[i])+'.csv'
	print('use', filename)
	voter = pd.read_csv('predict/'+filename, delimiter=',').values
	voters.append(voter[:, 1])
	if N_test == -1:
		N_test = voter.shape[0]
		test_list = voter[:, 0]

# N_test = 30
results = []
for i in range(0, N_test):
	# init
	voting_box = defaultdict(list)
	# collect votes
	for voter in voters:
		votes = voter[i].split(' ')
		votes.pop() # pop the last element which is empty string
		weight = 20
		for vote in votes:
			if voting_box[vote]==[]:
				voting_box[vote] = weight
			else:
				voting_box[vote] += weight
			weight-=1
	# count
	c = Counter(voting_box)
	result = ''
	have_new_whale = False
	new_whale_threshold = 21
	write_times = 0
	for candidate, N_votes in c.most_common(4):
		N_votes = N_votes/N_voter
		if N_votes<new_whale_threshold and not have_new_whale:
			result += 'new_whale ' + candidate+' '#+'['+str(N_votes)+'] '
			have_new_whale = True
		else:
			if not have_new_whale and write_times==3:
				result += candidate+' new_whale '
			else:
				result += candidate+' '#+'['+str(N_votes)+'] '
		write_times+=1
	# print(i, test_list[i], result)
	results.append(result[:-1]) # remove last 'space'
# write predict.csv
csv_content = np.concatenate([np.array(test_list).reshape(-1, 1), np.array(results).reshape(-1, 1)], axis=1)
pd.DataFrame(csv_content, columns=['Image', 'Id']).to_csv('predict/final_predict_train.csv', index=False)

			