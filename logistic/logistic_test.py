# logistic
import numpy as np
from matplotlib import pyplot as plt


def LoadData(filename):
    fr = open(filename)
    feature = []
    label = []
    while True:
        line = fr.readline()
        if line == '':
            break
        else:
            line_str = line.strip().split('\t')
            feature.append([float(x) for x in line_str[:-1]])
            label.append(int(line_str[-1]))
    return feature,label

def PlotData(feature,label):
    for i in range(4):
        plt.subplot(2,2,i+1)
        time = (i+1)*1
        x0 = feature[label == 0,1]
        x1 = feature[label == 0,2]
        plt.scatter(x0,x1,c='y',marker='>',label=0)
        x0 = feature[label==1,1]
        x1 = feature[label==1,2]
        plt.scatter(x0,x1,c='r',marker='x',label=1)
        plt.legend('upper left')
        plt.xlabel('feature[0]')
        plt.ylabel('feature[1]')
        plt.title('time'+str(time))
        weight = Gradientdecent(feature,label,time)
        # print(weight)
        x0 = [-3,5,0.1]
        x1 = (-weight[0]-weight[1]*x0)/weight[2]
        plt.plot(x0,x1)

    plt.show()


def Sigmoid(x):
    y = 1.0/(1+np.exp(x))
    return y

def Gradientdecent(feature,label,learn_step=0.001,time=100):
    label = label.reshape(len(label),1)
    m,n=np.shape(feature)
    weight = np.zeros((n,1)) 
    for i in range(time):
        h = feature.dot(weight)
        out = Sigmoid(h)
        error = label - out
        dw = feature.T.dot(error)
        weight += -learn_step*dw
    return weight




def Main():
    filename = u'F:/kaggle/codelib/logistic/testSet.txt'
    feature,label = LoadData(filename)
    # print(feature)
    w0 = np.ones((len(feature),1))
    feature = np.c_[w0,feature]
    # print(feature)
    feature=np.array(feature)
    label = np.array(label)
    PlotData(feature,label)


Main()