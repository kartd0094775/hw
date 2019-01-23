import numpy as np
import matplotlib.pyplot as plt
import operator
import csv
import math
import sys
with open('./train.csv', 'r') as f:
    train_file = f.readlines()
    f.close
with open('./test.csv', 'r') as f:
    test_file = f.readlines()
    f.close

def test(model, input_data): 
    w = model['w']
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
            print(output_id)
        output.write(output_id +','+str(output_value) +'\n')
        
    
 


# In[16]:


# ydata = b + w * xdata
lr = 1.0 # learning_rate
a = 1.0

class LinearRegression:
    
    def __init__(self, train, slot_number, feature_number):
        self.slot_number = slot_number
        self.feature_number = feature_number
        self.x = train['x']
        self.y = train['y']
        
    def stochasticGradientDescent(self, training, iteration):
        x_data = training['x']
        y_data = training['y']
        
        b = 0.0
        w = [0] * len(x_data[0])
        
        # Adagrad
        lr_b = 0.0
        lr_w = 0.0
        
        n = 0
        for i in range(iteration):

            regularizer = np.multiply(a, w)
            b_grad =  - 2.0 * (y_data[n] - b - np.dot(w, x_data[n]))*1.0 
            w_grad =  - np.multiply(2.0,(y_data[n] - b - np.dot(w, x_data[n])), x_data[n])

            lr_b = lr_b + b_grad ** 2
            lr_w = lr_w + w_grad ** 2
            # Update parameters
            b = b - lr/np.sqrt(lr_b) * b_grad
            w = w - lr/np.sqrt(lr_w) * w_grad
            
            n += 1
            if n % len(x_data) == 0:
                n = 0
        return {'b': b, 'w': w}
    def closedFormSol(self, training_set):
        x = training_set['x']
        y = training_set['y']
        
        x_pinv = np.linalg.pinv(x)
        w = np.dot(x_pinv, y)
        
        return {'w': w}
    def gradientDescent(self, training, iteration, lr=1.0, a=1.0):
        x = training['x']
        y = training['y']
        
        x_t = x.transpose()
        s_gra = np.zeros(len(x[0]))
        w = np.zeros(len(x[0]))
        
        for i in range(iteration):
            hypo = np.dot(x,w)
            loss = hypo - y + np.dot(w, w) * a / len(x[0])
            cost = np.sum(loss**2) / len(x)
            cost_rms = math.sqrt(cost)
            gra = np.add(np.dot(x_t, loss), np.multiply(a/len(x[0]),w))
            s_gra += gra**2.0  # adagrad
            ada = np.sqrt(s_gra)
            w = w - lr * gra/ada
        return {'w': w}
    def evaluation(self, model, testing_set):
        x_data = testing_set['x']
        y_data = testing_set['y']
        w = model['w']
        error = 0
        for i in range(len(x_data)):
            error += (y_data[i] -  np.dot(w, x_data[i])) ** 2
        return np.sqrt(error/len(x_data))
    def validation(self, iteration, mode='gd' , lr = 1.0, a=1.0):
        slot_size = int(len(self.x) / self.slot_number)
        error = 0.0
        err = []
        validation_x = []
        validation_y = []
        for i in range(self.slot_number):
            validation_x.append(self.x[i * slot_size: (i+1) * slot_size])
            validation_y.append(self.y[i * slot_size: (i+1) * slot_size])
            
        for i in range(self.slot_number):
            temp_x = np.array([]).reshape(0, self.feature_number)
            temp_y = np.array([])
            for j in range(self.slot_number):
                if i != j:
                    temp_x = np.concatenate((temp_x, validation_x[j]), axis=0)
                    temp_y = np.concatenate((temp_y, validation_y[j]), axis=0)
                        
            testing_set = {
                'x': validation_x[i],
                'y': validation_y[i]
            }
            training_set = {
                'x': temp_x,
                'y': temp_y
            }
            
            if mode == 'gd':
                result = self.gradientDescent(training_set, iteration, lr, a)
            elif mode == 'sgd':
                result = self.stochasticGradientDescent(training_set, iteration)
            elif mode == 'closed':
                result = self.closedFormSol(training_set)
            err.append(self.evaluation(result, testing_set))
            error += err[i]
        print('root mean squre: ' + str(error/self.slot_number))
        return error/self.slot_number
    def train(self, iteration, mode='gd', lr=1.0, a=1.0):
        training_set = {
            'x': self.x,
            'y': self.y
        }
        if mode == 'gd':
            result = self.gradientDescent(training_set, iteration, lr, a)
        elif mode =='sgd':
            result = self.stochasticGradientDescent(training_set, iteration)
        elif mode == 'closed':
            result = self.closedFormSol(training_set)
        return result


# In[17]:


def generateTestData():
    test_x = []
    n_row = 0
    text = open('./test.csv', 'r', encoding='big5') 
    row = csv.reader(text , delimiter= ",")
    
    for r in row:
        if n_row %18 == 0:
            test_x.append([])
            test_x[n_row//18].append(float(r[-1]))
        else :
            if n_row%18 != 10:
                test_x[n_row//18].append(float(r[-1]))
        n_row = n_row+1
    text.close()
    test_x = np.array(test_x)
    #test_x = np.concatenate((test_x,test_x**2), axis=1)
    # 增加平方項
    test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)
    # 增加bias項  
    return test_x

def generateTrainData():
    data = []    
    #一個維度儲存一種污染物的資訊
    for i in range(18):
        data.append([])

    n_row = 0
    text = open('train.csv', 'r', encoding='big5') 
    row = csv.reader(text , delimiter=",")
    for r in row:
        # 第0列為header沒有資訊
        if n_row != 0:
            # 每一列只有第3-27格有值(1天內24小時的數值)
            for i in range(3,27):
                if r[i] != "NR":
                    data[(n_row-1)%18].append(float(r[i]))
                else:
                    data[(n_row-1)%18].append(float(0))	
        n_row = n_row+1
    text.close()
    x = []
    y = []

    for i in range(12):
        # 一個月取連續10小時的data可以有471筆
        for j in range(479):
            pm25 = data[9][480*i+j+1]
            if (pm25 <= 0): continue
            # 總共有18種污染物
            temp = []
            for t in range(18):
                if t==10: continue
                temp.append(data[t][480*i+j] )
            x.append(temp)
            y.append(pm25)
    x = np.array(x)
    y = np.array(y)

    #x = np.concatenate((x,x**2), axis=1)
    # 增加平方項

    x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)
    # 增加bias項
    return {'x': x, 'y': y}


# In[ ]:


if __name__ == '__main__':
    train_data = generateTrainData()
    test_data = generateTestData()
    demo = LinearRegression({'x': train_data['x'], 'y': np.array(train_data['y'])}, 12, 1 * 17 + 1)
    result = demo.train(0, 'closed', 10000, 0.4)
    np.save('hw1_best.npy',result['w'])
    output = test(result, test_data)
    outputFile(output, test_file, 'hw1_best.csv')
