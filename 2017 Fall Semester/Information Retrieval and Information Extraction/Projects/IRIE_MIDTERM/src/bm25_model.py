
# coding: utf-8

# In[4]:


import json
from pprint import pprint
import nltk
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

import math
from six import iteritems
from six.moves import xrange


# BM25 parameters.
PARAM_K1 = 1.0
PARAM_B = 0.75
EPSILON = 0.25


class BM25(object):

    def __init__(self, corpus):
        self.corpus_size = len(corpus)
        self.avgdl = sum(float(len(x)) for x in corpus) / self.corpus_size
        self.corpus = corpus
        self.f = []
        self.df = {} # document-frequency dict
        self.idf = {} # inverted document-frequency dict
        self.initialize()

    def initialize(self):
        
        # pre-processing for each document
        for document in self.corpus:
            frequencies = {}
            
            # To calculating the the frequencies of the index terms for each document
            for word in document:
                                    
                # The word hasn't appeared before
                if word not in frequencies:
                    frequencies[word] = 0
                    
                frequencies[word] += 1
            self.f.append(frequencies)
            
            # Initialized the document-frequency list
            for word, freq in iteritems(frequencies):
                if word not in self.df:
                    self.df[word] = 0
                self.df[word] += 1
        # BM1: log (Ni - f + 0.5) / (f + 0.5) 
        for word, freq in iteritems(self.df):
            self.idf[word] = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)

    def get_score(self, query, index, average_idf):
        score = 0
        
        for word in query:
            
            # ther filter of the stop words
            if word in stop:
                continue
            # the word in the query but not in the document
            if word not in self.f[index]:
                continue
                
            idf = self.idf[word] if self.idf[word] >= 0 else EPSILON * average_idf
            #idf = self.idf[word] if self.idf[word] >= 0 else 0
            score += (idf * self.f[index][word] * (PARAM_K1 + 1)
                      / (self.f[index][word] + PARAM_K1 * (1 - PARAM_B + PARAM_B * len(self.corpus[index]) / self.avgdl)))
        return score
    
    # given query to calculating the score of all the document
    def get_scores(self, query, average_idf):
        scores = []
        for index in xrange(self.corpus_size):
            score = self.get_score(query, index, average_idf)
            scores.append(score)
        return scores


# In[6]:


from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
from nltk.stem import PorterStemmer
ps = PorterStemmer()
import json

with open('DBdoc.json') as dbdoc:
    db = json.load(dbdoc)

corpus = list()

for i in range(len(db)):
    if (db[i]['abstract'] is not None):
        temp = tokenizer.tokenize(db[i]['abstract'])
        document = []
        for word in temp:
            if word not in stop and len(word) > 3:
                document.append(ps.stem(word).lower())     
    else:
        document = list()
        
    for term in db[i]['entity'].split('_'):
        document.append(ps.stem(term).lower())
    corpus.append(document)
        

bm25 = BM25(corpus)

input_query = open('queries-v2.txt', 'r', encoding='UTF-8')
content = input_query.readlines()
output_file = open('bm25', 'w');   
count = 0 
for line in content:
    count += 1
    if (count % 40) == 0:
        print(count)
    
    temp = line.split('\t')

    sentence = temp[1][:-1]
            
    query = []
    for word in sentence.split():
        if word not in stop:
            query.append(ps.stem(word).lower())
        
    scores = bm25.get_scores(query, 0)
        
    rank = 0
    for i in range(100):
        rank = rank + 1
        score = max(scores)
        index =scores.index(score)
        output_file.write(temp[0] + "\tQ0\t<dbpedia:" + db[index]['entity'] + ">\t" + str(rank) + "\t" + str(score) + "\tSTANDARD\n")
        scores[index] = -1
output_file.close()

