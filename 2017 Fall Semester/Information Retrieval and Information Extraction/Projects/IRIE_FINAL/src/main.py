
# coding: utf-8
import re
from six import iteritems
import xgboost as xgb
from math import log
import math
import numpy as np

with open('train.txt', encoding='utf-8') as f:
    train_data = f.readlines()
    f.close
with open('test.txt', encoding='utf-8') as f:
    test_data = f.readlines()
    f.close
with open('Dream_of_the_Red_Chamber_seg.txt', encoding='utf-8') as f:
    corpus_seg = f.readlines()
    f.close
with open('Dream_of_the_Red_Chamber.txt', encoding='utf-8') as f:
    corpus = f.readlines()
    f.close

def tf_idf(corpus,term_dic): #Create function (arg1,arg2....)
    # treat every paragraph as a document : N=num_of_para
    num_of_para=len(corpus)
    temp_dic={}
    for term in term_dic:
        doc_freq=0
        term_freq=0
        for paragraph in corpus:
            if (paragraph.find(term)>=0):
                doc_freq+=1
                term_freq+=paragraph.count(term)     
        if doc_freq==0:
            weight=0
        else:
            weight=(1 + math.log10(term_freq)) * math.log10(num_of_para/doc_freq)
        temp_dic[term] = weight
    return temp_dic


def isGeneralName(name):
    if len(name) == 2:
        return False
    else:
        subname = name[1:]
        if subname in GENERAL_NAME:
            return True

def isContainName(content, general_tag, name):
    if name in content:
        return True
    elif(general_tag):
        first_name = name[0]
        last_name = name[1:]
        if first_name in content and last_name in content:
            return True
    return False

class FeatureExtractor:
    
    def __init__(self, per1, per2, rel, file, feature_type):
        self.per1 = per1
        self.per2 = per2
        self.rel = rel
        self.file = file
        self.feature_type = feature_type
        self.word_vector = {}
        self.features = {}
        self.initialize()
    def initialize(self):
        
        # FEATURE0 - RELATION
        self.features['關係'] = 12
        
        # Intialized the word vector
        for term in TERM_LIST:
            self.word_vector[term] = 0
            
        # Added Person Number(e.g. 曹雪芹: 1, 賈寶玉: 2...)
        p1 =  PERSON[self.per2] if PERSON[self.per1] > PERSON[self.per2] else PERSON[self.per1]
        p2 =  PERSON[self.per1] if PERSON[self.per1] > PERSON[self.per2] else PERSON[self.per2]
        
        # FEATURE - Per_1
        # FEATURE - Per_2
        self.features['角色一'] = p1
        self.features['角色二'] = p2
            
        #FEATURE - Term_Group
        for feature, value in iteritems(GROUPS):
            self.features[feature] = 0
            
        # FEATURE - Last_name
        # Determine the last name is same or not
        if self.per1[0] == self.per2[0]:
            self.features['姓'] = 100
        else:
            self.features['姓'] = 0
    def extract(self, content, priority):
        weight = priority
        tokens = content.split()

        for token in tokens:
            
            # Useless Term
            if '_P' in token:
                continue
            elif '_DE' in token:
                continue
            elif '_T' in token:
                continue
            elif '_SHI' in token:
                continue
            
            term = token.split('_')[0]
            for feature, rule in iteritems(GROUPS):
                weight = priority
                if term in rule:
                    self.features[feature] += 1 * weight                
            self.word_vector[term] += 1 * priority
    def save(self):
        word_freq = ''
        feature_str = ''
        
        for word, freq in iteritems(self.word_vector):
            word_freq = word_freq + str(freq) + ','
            
        for feature, score in iteritems(self.features):
            feature_str = feature_str + str(score) + ','
        
        if (self.feature_type == 'TERMnGROUP'):
            self.file.write(str(RELATION[self.rel]) + ',' + word_freq + "," + feature_str[:-1] + '\n')
        elif (self.feature_type == 'GROUP'):
            self.file.write(str(RELATION[self.rel]) + ',' + feature_str[:-1] + '\n')
        elif (self.feature_type == 'TERM'):
            self.file.write(str(RELATION[self.rel]) + ',' + word_freq[:-1] + '\n')
        else:
            raise ValueError('Wrong Feature Type!')
        
        
class Preprocessor:
    
    def __init__(self, data, corpus, filename):
        self.file = open(filename, 'w')
        self.data = data
        self.extracted = [None] * len(data)
        self.corpus = corpus
        self.vector = {}
        
    def close(self):
        self.file.close()
        
    def transform(self, feature_type):
        index = 0
        for row in self.data:
            extracted = []
            extractor = FeatureExtractor(row[0], row[1], row[2], self.file, feature_type)
            
            per1 = row[0]
            per2 = row[1]
            per1_general = isGeneralName(per1)
            per2_general = isGeneralName(per2)
            
            tag = False
            
            # Sentence
            for paragraph in self.corpus:
                sentences = re.split('，|。|？|！|；', paragraph)
                for i in range(len(sentences)):
                    if isContainName(sentences[i], per1_general, per1) and isContainName(sentences[i], per2_general, per2):
                        extractor.extract(sentences[i], PRIORITY[0])
                        extracted.append('S: ' + sentences[i])
                        if (tag == False):
                            tag = True
            
            # Context
            if tag == False:
                for paragraph in self.corpus:
                    sentences = re.split('，|。|？|！|；', paragraph)
                    for i in range(len(sentences)-2):
                        context = sentences[i] + sentences[i+1] + sentences[i+2]
                        if isContainName(context, per1_general, per1) and isContainName(context, per2_general, per2):
                            extractor.extract(context, PRIORITY[1])
                            extracted.append('C: ' + context)
                            if (tag == False):
                                tag = True
            # Otherwise
            if tag == False:
                temp = ['', '']
                for paragraph in self.corpus:
                    sentences = re.split('，|。|？|！|；', paragraph)
                    for sentence in sentences:
                        if isContainName(sentence, per1_general, per1) and temp[0] == '':
                            temp[0] = sentence
                        if isContainName(sentence, per2_general, per2) and temp[1] == '':
                            temp[1] = sentence
                    if temp[0] != '' and temp[1] != '':
                        extractor.extract(temp[0] + ' ' + temp[1], PRIORITY[3])
                        extracted.append('O: ' + temp[0] + ' ' + temp[1])
                        if tag == False:
                            tag = True
                        break
                        
            # Otherwise
            if tag == False:
                temp = ['', '']
                for paragraph in CORPUS:
                    sentences = re.split('，|。|？|！|；', paragraph)
                    for sentence in sentences:
                        if per1 in sentence and temp[0] == '':
                            temp[0] = sentence
                        if per2 in sentence and temp[1] == '':
                            temp[1] = sentence
                    if temp[0] != '' and temp[1] != '':
                        extracted_content = temp[0] + ' ' + temp[1]
                        extracted.append('R: ' + extracted_content)
                        break
                
            self.extracted[index] = extracted
            index +=1
            extractor.save()


# In[4]:


def generateFeatureFile(train, test, corpus, feature_type):
    # Training Data & Testing Data Transformation
    pre1 = Preprocessor(train, corpus, 'ftrain.txt')
    pre1.transform(feature_type)
    pre1.close()

    pre2 = Preprocessor(test, corpus, 'ftest.txt')
    pre2.transform(feature_type)
    pre2.close()


# In[5]:


def xgboostTraining(depth, objective):
    dtrain = xgb.DMatrix('ftrain.txt')
    dtest = xgb.DMatrix('ftest.txt')
    # specify parameters via map
    param = {}
    # use softmax multi-class classification
    param['objective'] = 'multi:soft' + objective
    # scale weight of positive examples
    param['eta'] = 0.1126
    param['max_depth'] = depth
    param['silent'] = 1
    param['num_class'] = 12
    num_round = 2
    bst = xgb.train(param, dtrain, num_round)
    # make prediction
    bst.save_model('temp.txt')
    bst = xgb.Booster(param)
    bst.load_model('temp.txt')
    preds = bst.predict(dtest)
    return preds


# In[6]:


def learningBasedEvaluation(depth, feature_type):
    #generateFeatureFile(TRAIN, TEST, corpus_seg, feature_type)
    preds = xgboostTraining(depth, 'max')
    
    point = 0
    for i in range(len(TEST)):
        if (preds[i] == RELATION[TEST[i][2]]):
            point += 1
    print(point / len(TEST))


# In[7]:


def mergedEvaluation(xgboost_preds, rule_based_result, threshold):
    error = 0
    preds = xgboost_preds
    result = rule_based_result
    
    for i in range(112):
        prob = np.amax(preds[i])
        label = preds[i].tolist().index(prob)
        test_label = TEST[i][2]
        if prob < threshold:
            if result[i] != test_label:
                error += 1;
        else:
            if (label != RELATION[test_label]):
                error += 1;
    return(1 - error/112)


# In[8]:


class Person:
    def __init__(self, name):
        self.name = name;
        self.alias = '';
        self.weight = 0
        self.lastName = name[0];
        self.gender = 0 # Unknown: 0, Male: 1, Female: 2
        self.hasMaleHead = False
        self.hasAliasName = False
        self.hasServantFeature = False
        self.hasLastName = False
        self.initialize()
    def initialize(self):
        # 冠夫姓
        for f in featureDic['夫姓']:
            if (f in self.name):
                self.hasMaleHead = True
                break
        # 女性
        for f in featureDic['女性']:
            if (f in self.name):
                self.gender = 2
                break
                
        # 常用名字稱呼
        if (len(self.name) == 3 and self.name[1:] not in GENERAL_NAME):
            self.alias = self.name[1:]
            self.hasAliasName = True
            
        # 僕人特徵
        if (self.name[-1] in ['奴', '兒']):
            self.hasServantFeature = True
        
        #姓
        if (self.name[0] in ['賈', '薛', '史', '林', '王', '尤', '李', '傅', '裘', '劉']):
                self.hasLastName = True
    def compareLastName(self, p2):
        if self.lastName == p2.lastName:
            return True
        return False
    def validName(self):
        if (self.hasAliasName):
            return self.alias
        return self.name
    def isFemale(self):
        if (self.gender == 2):
            return True
        return False
    def isServant(self):
        if (self.hasServantFeature):
            return True
        return False
    def isImportant(self):
        if (len(self.name) == 3 or self.hasLastName):
            return True
        return False


# In[9]:


def ruleBase(testData, weight_lastName):

    preFile = open('Segpreprocess.txt','w') 

    rightRelation=[]
    judgeRelation=[]
    
    # search corpus by two entities and relationship
    for row in testData:
        oneLine=[]
        oneSentence=[]
        threeSentences=[]
        oneParagraph=[]
        rightRelation.append(row[2])
        
        # Rule: 其中一個是地方就不用做feature list判斷了，一定是居處
        if (row[0] in featureDic['地點']) or (row[1] in featureDic['地點']):
            judgeRelation.append('居處')        
            continue
            
        per1 = Person(row[0])
        per2 = Person(row[1])
        
        # Rule: 兩人之中有一人為僕人
        if per1.isServant() != per2.isServant():
            judgeRelation.append('主僕')
            continue
        
        per1.weight = term_weight_dic[per1.name] if per1.name in term_weight_dic else 0
        per2.weight = term_weight_dic[per2.name] if per2.name in term_weight_dic else 0
        
        
        # Rule: 三個字人名如果有姓，去掉比較好找。若最後一個字是「娘」代表是姨娘，姓不可以省略
        entity1 = per1.validName()
        entity2 = per2.validName()

            
        preFile.write(row[0]+' '+row[1]+' '+row[2]+'\n')
        for paragraph in corpus:
            if ((entity1 in paragraph) and (entity2 in paragraph)):
                inLine = False
                inSentence = False
                inThreeSentences = False
                lines = re.split('[，；。？！]', paragraph)
                for line in lines:
                    if ((entity1 in line) and (entity2 in line)):
                        inLine = True
                        oneLine.append(line)
                        preFile.write("LINE:"+line+'\n')

                sentences = re.split('[。？！]', paragraph)
                thrSentences = []

                for sentence in sentences:
                    idx=sentences.index(sentence)
                    # create 3-sentences group
                    if idx>1:
                        thrSentences.append(sentences[idx-2]+"。"+sentences[idx-1]+"。"+sentences[idx])#中間標點統一以。代替
                    # judge if in the list
                    if ((entity1 in sentence) and (entity2 in sentence)):
                        inSentence = True
                        # 不能跟前面重複，中間沒逗點再加
                        commaLoc=[m.start() for m in re.finditer('[，；]', sentence)] # get all locations of '，；' 
                        hasCommaBetween = False
                        for cLoc in commaLoc:#暫不考慮同一人名一句話出現兩次的特例
                            a = sentence.find(row[0])
                            b = sentence.find(row[1])
                            if ((a<cLoc and cLoc<b) or (b<cLoc and cLoc<a)):
                                hasCommaBetween = True
                                break
                        if (hasCommaBetween == True):
                            oneSentence.append(sentence)
                            preFile.write("SENTENCE:"+sentence+'\n')

                for context in thrSentences:
                    if ((entity1 in context) and (entity2 in context)):
                        inThreeSentences = True
                        # 不能跟前面重複，中間沒。再加
                        periodLoc=[m.start() for m in re.finditer('[。]', context)] # get all locations of '。' 

                        hasPeriodBetween = False
                        for pLoc in periodLoc:#暫不考慮同一人名一句話出現兩次的特例
                            a = context.find(entity1)
                            b = context.find(entity2)
                            if ((a<pLoc and pLoc<b) or (b<pLoc and pLoc<a)):
                                hasPeriodBetween = True
                                break
                        if (hasPeriodBetween == True):
                            threeSentences.append(context)
                            preFile.write("CONTEXT:"+context+'\n')


                if not (inLine or inSentence or inThreeSentences):
                    oneParagraph.append(paragraph)
                    preFile.write("PARAGRAPH:"+paragraph+'\n')

        # create a dictionary to store appear phrase weight            
        term_weight_vector={}
        for line in oneLine:
            tempLine = line
            for term in term_dic:
                if term in tempLine:            
                    if term not in term_weight_vector:
                        term_weight_vector[term]=0
                    term_weight_vector[term] += tempLine.count(term) * 16

        for context in threeSentences:
            tempContext = context
            for term in term_dic:
                if term in tempContext:            
                    if term not in term_weight_vector:
                        term_weight_vector[term]=0
                    term_weight_vector[term] += tempContext.count(term) * 4
                
        for sentence in oneSentence:
            tempSentence = sentence
            for term in term_dic:
                if term in tempSentence:            
                    if term not in term_weight_vector:
                        term_weight_vector[term]=0
                    term_weight_vector[term] += tempSentence.count(term) * 2
                
        for paragraph in oneParagraph:
            tempParagraph = paragraph
            for term in term_dic:
                if term in tempParagraph:            
                    if term not in term_weight_vector:
                        term_weight_vector[term]=0
                    term_weight_vector[term] += tempParagraph.count(term) * 1
                
        # create feature list that symbolize different relationship
        featureList=[0]*12
        for term in term_weight_vector:
            if term in featureDic['婚配']:
                featureList[relationDic['夫妻']]+=term_weight_vector[term]*term_weight_dic[term]
            elif term in featureDic['直系']:
                featureList[relationDic['父子']]+=term_weight_vector[term]*term_weight_dic[term]
                featureList[relationDic['父女']]+=term_weight_vector[term]*term_weight_dic[term]
                featureList[relationDic['母子']]+=term_weight_vector[term]*term_weight_dic[term]
                featureList[relationDic['母女']]+=term_weight_vector[term]*term_weight_dic[term]
            elif term in featureDic['尊卑']:
                featureList[relationDic['祖孫']]+=term_weight_vector[term]*term_weight_dic[term]
                featureList[relationDic['主僕']]+=term_weight_vector[term]*term_weight_dic[term]
                featureList[relationDic['夫妻']]+=term_weight_vector[term]*term_weight_dic[term]
            elif term in featureDic['旁系']:
                featureList[relationDic['兄弟姊妹']]+=term_weight_vector[term]*term_weight_dic[term]
            elif term in featureDic['手足']:
                featureList[relationDic['兄弟姊妹']]+=term_weight_vector[term]*term_weight_dic[term]
                featureList[relationDic['主僕']]+=term_weight_vector[term]*term_weight_dic[term]
            elif term in featureDic['主僕']:
                featureList[relationDic['主僕']]+=term_weight_vector[term]*term_weight_dic[term]
            elif term in featureDic['命令']:
                featureList[relationDic['夫妻']]+=term_weight_vector[term]*term_weight_dic[term]
                featureList[relationDic['主僕']]+=term_weight_vector[term]*term_weight_dic[term]
                featureList[relationDic['父子']]+=term_weight_vector[term]*term_weight_dic[term]
                featureList[relationDic['父女']]+=term_weight_vector[term]*term_weight_dic[term]
                featureList[relationDic['母子']]+=term_weight_vector[term]*term_weight_dic[term]
                featureList[relationDic['母女']]+=term_weight_vector[term]*term_weight_dic[term]
            elif term in featureDic['遠親']:
                featureList[relationDic['遠親']]+=term_weight_vector[term]*term_weight_dic[term]
            elif term in featureDic['師徒']:
                featureList[relationDic['師徒']]+=term_weight_vector[term]*term_weight_dic[term]


        # 同姓：不會是主僕（僕人通常是暱稱），且更有可能是父子、父女、祖孫、兄弟姊妹、姑叔舅姨甥侄、遠親
        if per1.compareLastName(per2):
            featureList[relationDic['主僕']] = 0
            featureList[relationDic['遠親']] *= weight_lastName
            featureList[relationDic['父子']] *= weight_lastName
            featureList[relationDic['父女']] *= weight_lastName
            featureList[relationDic['祖孫']] *= weight_lastName
            featureList[relationDic['兄弟姊妹']] *= weight_lastName
            featureList[relationDic['姑叔舅姨甥侄']] *= weight_lastName

            # 若冠夫姓或父姓，同姓仍有可能是母子或母女
            if not per1.hasMaleHead and not per2.hasMaleHead:
                featureList[relationDic['母子']]= 0
                featureList[relationDic['母女']]= 0
            else:
                featureList[relationDic['母子']] *= weight_lastName
                featureList[relationDic['母女']] *= weight_lastName


        # 有女性就不會是父子、師徒
        if per1.isFemale() or per2.isFemale():
            featureList[relationDic['父子']] = 0
            featureList[relationDic['師徒']] = 0


        sorted_featureList = sorted(featureList)

        key_list = list(relationDic.keys())
        value_list = list(relationDic.values())
        
        # 無特徵值
        if sorted_featureList[-1] == 0:
            #print('===')
            #print(row)
            # 同性
            if (per1.compareLastName(per2)):
                rel = abs(per1.weight - per2.weight)
                #print(rel)
                #兩者關係不親
                if rel > 1:
                    #print(row[2] + ':' + '遠親')
                    judgeRelation.append('遠親')
                    continue
                #兩者關係較親密
                else:
                    #print(row[2] + ':' + '祖孫')
                    judgeRelation.append('祖孫')
                    continue
            elif (per1.isImportant() != per2.isImportant()):
                #print(row[2] + ':' + '主僕')
                judgeRelation.append('主僕')
                continue
        result = ''
        # 相同的狀況
        if (sorted_featureList[-1] == sorted_featureList[-2]):
            res1 = featureList.index(sorted_featureList[-1])
            res2 = featureList.index(sorted_featureList[-2], res1+1)
            res1 = key_list[value_list.index(res1)]
            res2 = key_list[value_list.index(res2)]
            pairRel = abs(per1.weight - per2.weight)
            strongRel = ['夫妻', '父子', '父女', '母女', '母子']
            weakRel = ['遠親', '主僕']
            if ((pairRel > 1) and (res1 in weakRel) and (res2 not in weakRel)):
                result = res1
            elif((pairRel <= 1) and (res1 not in strongRel) and (res2 in strongRel)):
                result = res1
            else:
                result = res1

        else:
            res = featureList.index(sorted_featureList[-1])
            result = key_list[value_list[res]]

        judgeRelation.append(result)
    preFile.close()            

    rightRate=0
    for i in range(len(judgeRelation)):
        if judgeRelation[i]==rightRelation[i]:
            rightRate+=1/len(judgeRelation)
        #else: 
            #print(testData[i])
            #print(judgeRelation[i]+"="+rightRelation[i])    
    #print ("rightRate="+str(rightRate))
    return judgeRelation


# In[10]:

if __name__ == '__main__':
    # Test & Train Array Converter
    TRAIN = []
    index = 0
    for row in train_data:
        if index == 0:
            index += 1
            continue
        index += 1
        x = re.split('\t|\n', row)
        TRAIN.append([x[1], x[2], x[3]])
        
    TEST = []
    # ========= For Learning Base Global Constant ==================
    index = 0
    for row in test_data:
        if index == 0:
            index += 1
            continue
        index += 1
        x = re.split('\t|\n', row)
        TEST.append([x[1], x[2], x[3]])

    # Build The Word Vector
    TERM_LIST = []
    for paragraph in corpus_seg:
        tokens = paragraph.split()
        for token in tokens:
            if '_P' in token:
                continue
            #term = re.sub('_[A-Z | a-z | 0-9]*', '', token)
            term = token.split('_')[0]
            if term not in TERM_LIST:
                TERM_LIST.append(term)

    RELATION = {
        '祖孫': 0,
        '母子': 1,
        '母女': 2,
        '父子': 3,
        '父女': 4,
        '兄弟姊妹': 5,
        '夫妻': 6,
        '姑叔舅姨甥侄': 7,
        '遠親': 8,
        '主僕': 9,
        '師徒': 10,
        '居處': 11,
    }

    PERSON = {}
    index = 1
    for x in TRAIN:
        per1 = x[0]
        per2 = x[1]
        if per1 not in PERSON:
            PERSON[per1] = index
            index += 1
        if per2 not in PERSON:
            PERSON[per2] = index
            index += 1
    for x in TEST:
        per1 = x[0]
        per2 = x[1]
        if per1 not in PERSON:
            PERSON[per1] = index
            index += 1
        if per2 not in PERSON:
            PERSON[per2] = index
            index += 1
            
    GENERAL_NAME = [
        '婆子', '夫人', '大姐', '小姐', '嫂子', '姨娘',  '姨媽', '嬸娘', '嫂子', '老娘', '嬤嬤', '奶奶'
    ]

    Group1 = ['嫁','娶','婚','買','嫡夫','婦','嫡','妻','妾','連理','太太','夫妻']
    Group2 = ['喚作','取名','生','有了','得了','養','懷','爹','娘','父','母','女','女兒','子','孩','乳名','小名']
    Group3 = ['請','給','來','請安','磕頭','問好','跪','稟明','奉','喚來','叫','祖','奶','孫','老太太','帶','領']
    Group4 = ['長','次','大']
    Group5 = ['兄','哥','弟','姊','姐','妹']
    Group6 = ['姑','叔','舅','姨','甥','侄','親']
    Group7 = ['帶','領','教','徒','門生','師父']
    Group8 = ['主','僕','丫','丫頭','丫鬟','心腹','小的','下人','主僕']
    Group9 = ['使喚','謝','領','接','扇','差','命','遣','迎','打發','吩咐','喚','罵']

    GROUPS = {
        '婚配': Group1, 
        '直系': Group2, 
        '尊卑': Group3, 
        '旁系': Group4, 
        '手足': Group5, 
        '遠親': Group6, 
        '師徒': Group7, 
        '主僕': Group8,
        '命令': Group9
    }

    CORPUS = corpus

    PRIORITY = [8, 4, 2, 1]
    # ==============For Rule Base Global Constant=============
    # Build The Dictinary
    term_dic = {} # {term:代號,...}
    term_weight_dic = {}
    featureDic={}

                    
    # for row in TRAIN:
    #     if row[0] not in term_dic:
    #         term_dic[row[0]] = 'Character'
    #     if row[1] not in term_dic:
    #         term_dic[row[1]] = 'Character'
            
    # for row in TEST:
    #     if row[0] not in term_dic:
    #         term_dic[row[0]] = 'Character'
    #     if row[1] not in term_dic:
    #         term_dic[row[1]] = 'Character'
        
    for paragraph in corpus_seg:
        tokens = paragraph.split()
        for token in tokens:
            # Normal Norm
            if '_Na' in token:
                pair = token.split('_')
                if pair[0] not in term_dic:
                    term_dic[pair[0]]=pair[1]
            if '_Nb' in token:
                pair = token.split('_')
                if pair[0] not in term_dic:
                    term_dic[pair[0]]=pair[1]
            # Location
            elif '_Nc' in token:
                pair = token.split('_')
                if pair[0] not in term_dic:
                    term_dic[pair[0]]=pair[1]
            # Time
            elif '_Nd' in token:
                pair = token.split('_')
                if pair[0] not in term_dic:
                    term_dic[pair[0]]=pair[1]
            elif '_V' in token:
                pair = token.split('_')
                if pair[0] not in term_dic:
                    term_dic[pair[0]]=pair[1]                

                    
    term_weight_dic = tf_idf(corpus,term_dic)

    featureDic['婚配']=['嫁','娶','婚','嫡夫','婦','嫡','妻','妾','連理','太太','夫妻', '媳婦', '夫婦']
    featureDic['直系']=['喚作','取名','生','有了','得了','養','懷','爹','娘','父','母','兒','女','女兒','子','孩','乳名','小名']
    featureDic['尊卑']=['請','給','來','請安','磕頭','問好','跪','稟明','奉','喚來','叫','祖','奶','孫','老太太','帶','領']
    featureDic['旁系']=['長','次','大']
    featureDic['手足']=['兄','哥','弟','姊','姐','妹']
    featureDic['遠親']=['姑','叔','舅','姨','甥','侄','親']
    featureDic['師徒']=['帶','領','教','徒','門生','師父']
    featureDic['主僕']=['主','僕','丫','丫頭','丫鬟','心腹','小的','下人','主僕']
    featureDic['命令']=['使喚','謝','領','接','扇','差','命','遣','迎','打發','吩咐','喚','罵']
    featureDic['女性']=['嬤','母','姐','姊','妹','太','夫人','氏','娘','女','姑','姨']
    featureDic['夫姓']=['姐','母','娘','媽','奶','嬤']    # 若冠夫姓或父姓，可能會出現的稱呼
    featureDic['地點']=[]
    for key in term_dic:
        if 'Nc' in term_dic[key]:
            featureDic['地點'].append(key)

    relationDic={'祖孫':0, '母子':1, '母女':2, '父子':3, '父女':4, '兄弟姊妹':5,'夫妻':6,
                '姑叔舅姨甥侄':7,'遠親':8,'主僕':9, '師徒':10,'居處':11 }


    generateFeatureFile(TRAIN, TEST, corpus_seg, 'TERMnGROUP')
    xgboost_preds = xgboostTraining(2, 'prob')
    rule_based_result = ruleBase(TEST, 2) # param: Evaluted File & Last Name Weight
    result = mergedEvaluation(xgboost_preds, rule_based_result, 0.140624)
    print('Right Rate: ' + str(result))

