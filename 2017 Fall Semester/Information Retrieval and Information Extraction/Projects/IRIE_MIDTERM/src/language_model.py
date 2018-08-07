
# coding: utf-8

# In[1]:


import math
from six import iteritems
from six.moves import xrange
import operator
import json
from pprint import pprint
import nltk
from nltk.corpus import stopwords


lamda = 0.5

class LM(object):
    def __init__(self, corpus):
        self.corpus = corpus
        self.corpus_size = len(corpus)
        
        # frequencies
        self.f = list()
        
        #length
        self.l = list()
        
        # probilities
        self.p = list()
        
        # collection-frequencies
        self.cf = dict()
        
        # collection-probilities
        self.cp = dict()
        
        # collection-length
        self.cl = 0
        self.initialize()
        
    def initialize(self):
        for document in self.corpus:
            frequencies = {}
            probabilities = {}
            length = 0
            # calculating the frequencies of the word for each document
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.f.append(frequencies)

            # calculating the probilities(freq/document_len) and the frequencies of the word in the collection
            for word, freq in iteritems(frequencies):
                probabilities[word] = freq / len(document) 
                if word not in self.cf:
                    self.cf[word] = 0
                self.cf[word] += freq
                length += freq
            
            self.l.append(length)
            self.p.append(probabilities)
            
            # summing up the lenght of the document
            self.cl += len(document)
            
        # calculating the probilities of the word in the collection
        for word, freq in iteritems(self.cf):
            self.cp[word] = freq/self.cl
           
        
    def smoothing(self, word, Index):
        mode = 'JM'
        
        # Jelinek-Mercer Method
        if (mode == 'JM'):
            lamda = 0.9
            return ((1 - lamda) * self.f[Index][word] / self.l[Index]) + (lamda * self.cf[word] / self.cl)
        
        # Dirichlet Smoothing
        elif(mode == 'D'):
            lamda = 0.5 
            return (self.f[Index][word] + lamda * (self.cf[word] / self.cl)) / (self.l[Index] + lamda)
            
            
    def get_scores(self, query):
        scores = list()
        for Index in xrange(self.corpus_size):
            score = self.get_score_simple(query, Index)
            scores.append(score)
        return scores
    
    def get_score_multinomial(self, query, Index):
        # the probilities of all the index word
        word_prob_list = []
        
        #the word don't belong to corpus[index]
        other_word = list()
        
        # the sum of the probilities of the word in the corpus[index]
        sum_word_prob = 0
        # the sum of the collection probilties of the word in the corpus[index]
        sum_col_prob = 0
        
        for word in query:
            
            # filter the not existing word
            if word not in self.cp:
                continue
                
                
            if  word in self.f[Index]:  
            
                document_s_prob = self.smoothing(word, Index)
            
                word_prob_list.append(document_s_prob)
                sum_word_prob += document_s_prob
                sum_col_prob += self.cp[word]
            else:
                other_word.append(word)
                
        for word in other_word:
                alpha = ( 1 - sum_word_prob ) / ( 1 - sum_col_prob ) 
                word_prob_list.append(alpha * self.cp[word])

        return self.calculate_unigram_prob(word_prob_list)
    def get_score_simple(self, query, Index):
        word_prob = list()
        for word in query:
            
            # filter the not existing word
            if word not in self.cp:
                continue
            
            document_prob = self.p[Index][word] if word in self.p[Index] else 0 
            collection_prob = self.cp[word]
            result = document_prob * lamda + collection_prob * (1 - lamda)
            word_prob.append(result)
        return self.calculate_unigram_prob(word_prob)

    def query_expansion(self, query, documents_index, Size):
        frequencies = {}
        for index in documents_index:
            for word, frequency in iteritems(self.f[index]):
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += frequency
        
        for i in range(Size):
            max_term = max(frequencies.items(), key = operator.itemgetter(1))[0]
            if max_term not in query:
                query.append(max_term)
            frequencies[max_term] = -1
        return query
    def calculate_unigram_prob(self, probilities):
        score = 1.0 
        for prob in probilities:
            score = score * prob
        return score

stops = set(stopwords.words('english'))

# reading the documents of the dbpedia
with open('DBdoc.json') as dbdoc:
    db = json.load(dbdoc)
    


import json
from pprint import pprint
import nltk
from nltk.corpus import stopwords
stops = set(stopwords.words('english'))

from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))


with open('DBdoc.json') as dbdoc:
    db = json.load(dbdoc)

corpus = list()

for i in range(len(db)):
    if (db[i]['abstract'] is not None):
        temp = tokenizer.tokenize(db[i]['abstract'])
        document = []
        for word in temp:
            if word not in stop and len(word) > 3:
                document.append(word.lower())     
    else:
        document = list()
        
    for term in db[i]['entity'].split('_'):
        document.append(term.lower())
    corpus.append(document)
        
        
lm = LM(corpus)


input_file = open('queries-v2.txt', 'r', encoding='UTF-8')
content = input_file.readlines()
output_file = open('language', 'w', encoding='UTF-8')

for line in content:
    temp = line.split('\t')
    sentence = temp[1][:-1]
            
    query = []
    for word in sentence.split():
        if word not in stop:
            query.append(word.lower())
    query = []
    for word in sentence.split():
        if word not in stop:
            query.append(word.lower())

    scores = lm.get_scores(query)
#     expansion_list = list()
#     for i in range(10):
#         index = scores.index(max(scores))
#         expansion_list.append(index)
#         scores[index] = -1

#     index_term = lm.query_expansion(index_term, expansion_list, 5)
#     scores = lm.get_scores(index_term)
    
    rank = 0
    for i in range(100):
        rank += 1
        score = max(scores)
        index = scores.index(score)
        output_file.write(temp[0] +"\tQ0\t<dbpedia:" + db[index]['entity'] + ">\t" + str(rank) + "\t" + str(score) + "\tSTANDARD\n")
        scores[index] = -1
output_file.close()

