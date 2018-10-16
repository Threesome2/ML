#knn算法，机器学习实战源代码

import numpy as np
import operator

#创建简单的数据用例x，y
def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

#knn算法实现
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet  #tile()用于扩展矩阵
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5 #求得距离
    sortedDistIndicies = distances.argsort()  #返回排序的下标，从小到大   
    classCount={}          
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1 #用字典计数
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    #对字典排序，基于值，从大到小(逆序，reverse=True)
    return sortedClassCount[0][0]

#测试一下

# group,labels=createDataSet()

# pred=classify0([0,0],group,labels,3)

# print(pred)


#导入数据集
def file2matrix(filename):
    print(filename)
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    returnMat = np.zeros((numberOfLines,3))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return   
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

# #导入数据
# datingDataMat,datingLabels = file2matrix(u"F:\kaggle\codelib\knn\datingTestSet2.txt")
# print(datingDataMat[0:5])
# print(datingLabels[0:5])

#可视化数据集合
import matplotlib.pyplot as plt

def plot(x,y):
    label1 = np.where(y == 1)[0]
    plt.scatter(x[label1,0],x[label1,1],marker='x',color = 'r',label = 'didnt like=1')
    label2 = np.where(y == 2)[0]
    plt.scatter(x[label2,0],x[label2,1],marker='o',color = 'b',label = 'smallDoses=2')
    label3 = np.where(y == 3)[0]
    plt.scatter(x[label3,0],x[label3,1],marker='.',color = 'y',label = 'largeDoses=3')
    plt.xlabel('pilot distance')
    plt.ylabel('game time')
    plt.legend(loc = 'upper left')
    plt.show()
    
# plot(datingDataMat,np.array(datingLabels).reshape(-1,1)) 



#数值归一化

def autoNorm(dataSet): 
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m,1))
    normDataSet = normDataSet/np.tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.10      #hold out 10%,留作test集合
    datingDataMat,datingLabels = file2matrix(u"F:\kaggle\codelib\knn\datingTestSet2.txt")       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio) #前面的numTestVecs作为测试集，后面的作为训练集
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:] 
                                     ,datingLabels[numTestVecs:m],3)
        #print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))
    print(errorCount)

datingClassTest()



# result = ["didnt like","small dose","large dose"]
# input = np.array([[10000,10,0.5]])
# #一定记得使用训练集去autoNorm，并且同时作用于测试数据
# normMat, ranges, minVals = autoNorm(datingDataMat)

# pred = classify0((input-minVals)/ranges,normMat,datingLabels,3)
# print( pred)
# print("you will probablly like this person:",result[pred-1]) 


# from os import listdir

# def img2vector(filename):
#     returnVect = np.zeros((1,1024))
#     fr = open(filename)
#     for i in range(32): #32*32----->1*1024
#         lineStr = fr.readline()
#         for j in range(32):
#             returnVect[0,32*i+j] = int(lineStr[j])
#     return returnVect

# def handwritingClassTest():
#     hwLabels = []
#     trainingFileList = listdir(u'F:/kaggle/codelib/knn/trainingDigits')   #load the training set
#     m = len(trainingFileList)
#     trainingMat = np.zeros((m,1024))
#     for i in range(m):
#         fileNameStr = trainingFileList[i]
#         fileStr = fileNameStr.split('.')[0]     #take off .txt
#         classNumStr = int(fileStr.split('_')[0]) #获得y的类别
#         hwLabels.append(classNumStr)
#         trainingMat[i,:] = img2vector(u'F:\kaggle\codelib\knn\trainingDigits/%s' % fileNameStr)
#     testFileList = listdir(u'F:\kaggle\codelib\knn\testDigits')        #iterate through the test set
#     errorCount = 0.0
#     mTest = len(testFileList)
#     for i in range(mTest):
#         fileNameStr = testFileList[i]
#         fileStr = fileNameStr.split('.')[0]     #take off .txt
#         classNumStr = int(fileStr.split('_')[0])
#         vectorUnderTest = img2vector(u'F:\kaggle\codelib\knn\testDigits/%s' % fileNameStr)
#         classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3) 
#         #预测，不需要归一化，因为数值已经是0/1
#         #print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
#         if (classifierResult != classNumStr): errorCount += 1.0
#     print("\nthe total number of errors is: %d" % errorCount)
#     print("\nthe total error rate is: %f" % (errorCount/float(mTest))) 
    
# #测试一下
# handwritingClassTest()



