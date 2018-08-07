import json
from pprint import pprint
import nltk
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))

import math
from six import iteritems
from six.moves import xrange


class VSM(object):

    def __init__(self, corpus):
        self.corpus_size = len(corpus)
        self.avgdl = sum(float(len(x)) for x in corpus) / self.corpus_size
        self.corpus = corpus
        
        self.vectors = []
        self.length = []
        
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
            

        # log (Ni - f + 0.5) / (f + 0.5) 
        for word, freq in iteritems(self.df):
            self.idf[word] = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            
        for Index in range(self.corpus_size):
            frequency = self.f[Index]
            value = 0
            for word, freq in iteritems(frequency):
                weight = (1 + math.log(freq)) * (math.log(self.corpus_size) - self.idf[word])
                value += pow(weight, 2)
            self.length.append(value)
            
            
    def get_score(self, query, index, average_idf):
        score = 0

        frequency = {}
        for word in query:
            
            # ther filter of the stop words
            if word in stop:
                continue
            # the word in the query but not in the document
            if word not in self.f[index]:
                continue
                
            if word not in frequency:
                frequency[word] = 0
            frequency[word] += 1
            
        score = 0
        for word, freq in iteritems(frequency):
            weight_q = (1 + math.log(freq)) * self.idf[word]
            weight_d = (1 + math.log(self.f[index][word])) * self.idf[word]
            score += weight_q * weight_d 
            
        return score / self.length[index]
    
    # given query to calculating the score of all the document
    def get_scores(self, query, average_idf):
        scores = []
        for index in xrange(self.corpus_size):
            score = self.get_score(query, index, average_idf)
            scores.append(score)
        return scores




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
        

vsm = VSM(corpus)


input_file = open('queries-v2.txt', 'r', encoding='UTF-8')
content = input_file.readlines()
output_file = open('output', 'w', encoding='UTF-8')

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

    scores = vsm.get_scores(query, 0)

    rank = 0
    for i in range(100):
        rank += 1
        score = max(scores)
        index = scores.index(score)
        output_file.write(temp[0] +"\tQ0\t<dbpedia:" + db[index]['entity'] + ">\t" + str(rank) + "\t" + str(score) + "\tSTANDARD\n")
        scores[index] = -1
output_file.close()


