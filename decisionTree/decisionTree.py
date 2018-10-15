
# coding: utf-8

# In[1]:


#决策树算法

import math
#小的数据集
def createDataSet():
    dataSet = [[1, 1, 'maybe'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    featureNames = ['no surfacing','flippers'] #不浮出水面是否存活 ，有无脚蹼
    #change to discrete values
    return dataSet, featureNames

#计算信息熵,因为我们会利用最大信息增益的方法划分数据集-----看哪个特征划分使得，信息熵(数据无序度)减小的最多
def Entropy(dataSet):
    num = len(dataSet)
    labelCounts = {}
    for featVec in dataSet: # 统计每个类别的数量
        currentLabel = featVec[-1] #最后1列为键
        if currentLabel not in labelCounts.keys(): 
            labelCounts[currentLabel] = 0 #初始值=0
        labelCounts[currentLabel] += 1 #统计+1
    entropy = 0.0
    for key in labelCounts:  #
        prob = float(labelCounts[key])/num
        entropy -= prob * math.log(prob,2) #log base 2
    return entropy

#测试一下
myData,myFeatureNames = createDataSet()
print("the old dataset:\n",myData)

myEntropy = Entropy(myData)
print("my test entropy should be 0.97095 :\n",myEntropy)

#添加一类，mabey,yes,no   ====熵越高，表明数据越混乱
myData[0][-1] = "mabey"
print("the new dataset:\n",myData)
myEntropy = Entropy(myData)
print("my test entropy should be 1.37095 :\n",myEntropy)


# In[2]:


#划分数据集


#按照给定的特征axis,根据他的取值value，划分数据集，返回新的数据集合，少了1个特征（划分依据的那个特征axis）

def splitDataSet(dataSet, axis, value):
    returnDataSet = []
    for dataVec in dataSet:
        if dataVec[axis] == value:
            tempVec = dataVec[:axis]     #0--(axis-1)
            tempVec.extend(dataVec[axis+1:]) #(axis+1)--(-1) #所以减去了axis
            returnDataSet.append(tempVec) 
    return returnDataSet

#测试一下
myData,myFeatureNames = createDataSet()
# print('=============')
# print(splitDataSet(myData,0,1)) #axis = 0,且这个特征的值=1


# In[3]:


#看哪个特征划分使得，信息熵(数据无序度)减小的最多

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      #最后1列是类别
    baseEntropy = Entropy(dataSet)         #首先计算原始的信息熵
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):        #迭代所有的特征
        #将数据集中的第i个特征的值，放到一个list中
        featureList = [example[i] for example in dataSet]
        uniqueVal = set(featureList)       #用set去重
        newEntropy = 0.0
        for value in uniqueVal:  
            print(value,uniqueVal)
            subDataSet = splitDataSet(dataSet, i, value)#对第i个特征，针对某个值划分
            print(subDataSet)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * Entropy(subDataSet)   #累加信息熵  
        infoGain = baseEntropy - newEntropy     #计算这次划分的信息增益
        print('++++++++')
        print(newEntropy)
        if (infoGain > bestInfoGain):       
            bestInfoGain = infoGain         #替换好的值
            bestFeature = i
    return bestFeature                      #return 最好的划分特征下标i



# 测试下
index = chooseBestFeatureToSplit(myData)
print("best feature shoule be 0\n",index)


# In[4]:


import operator

#对字典排序，取得最大值:将多数的类别标签作为“叶子节点”的类别
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    #sorted(classCount.items(), key=lambda x:x[1], reverse=True) #对字典排序
    return sortedClassCount[0][0] #返回多数的那个类别

#测试
c = [1,1,1,0,0,2,2,2,2]
print("the majorityClassCount is 2\n",majorityCnt(c))


# In[5]:


#根据数据集合，创建一个完整的决策树

def createTree(dataSet,featureNames):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList): #如果所有的类都一样，第0类的个数==长度
        return classList[0]#stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
        return majorityCnt(classList) #如果所有的特征都被遍历用于划分数据
    bestFeature = chooseBestFeatureToSplit(dataSet)
    bestFeatureName = featureNames[bestFeature]
    myTree = {bestFeatureName:{}} #用字典存储树结构
    del(featureNames[bestFeature]) #从特征名称列表中删除这个bestFeatureName
    featureValues = [example[bestFeature] for example in dataSet]  #遍历最好的特征的取值，进行划分
    uniqueVals = set(featureValues)
    for value in uniqueVals:
        subNames = featureNames[:]  #copy all of featureName, 为了保留原有的featureName不被函数修改
        myTree[bestFeatureName][value] = createTree(splitDataSet(dataSet, bestFeature, value),subNames) #返回的是一个字典
    return myTree 



#测试一下 
myData,myFeatureNames = createDataSet()
myTree = createTree(myData,myFeatureNames)
print(myTree)
print(myFeatureNames)  #最后的list，会被改变(！！！因为函数参数是list,参数是按照引用的方式传递的)


# In[6]:


#测试算法：对于某个输入变量x，使用决策树，进行分类

def classify(inputTree,featureNames,testVec):
    firstKey = list(inputTree.keys())[0] #第一个key,根节点,k=某个特征a的名称
    secondDict = inputTree[firstKey] #第二个字典,key是特征a的所有取值
    featureIndex = featureNames.index(firstKey)
    key = testVec[featureIndex] 
    valueOfFeature = secondDict[key] #获得叶子节点的值，要么是“类别标签”，要么是key=某个特征b的“字典”
    if isinstance(valueOfFeature, dict): #判断实例是否是类型(tuple,dict,int,float) 
        classLabel = classify(valueOfFeature, featureNames, testVec) #是字典，继续递归
    else: classLabel = valueOfFeature #是类别标签，直接返回
    return classLabel

myData,myFeatureNames = createDataSet() #因为myFeatureNames在函数中，改变了，所以重新加载
class0 = classify(myTree,myFeatureNames,[1,0])
print("[1,0] should be classify as 'no'\n",class0)

class1 = classify(myTree,myFeatureNames,[1,1])
print("[1,1] should be classify as 'yes'\n",class1)


# In[7]:


#使用pickle模块保存决策树参数和结构

import pickle

def storeTree(inputTree,filename):
    fw = open(filename,'wb+')
    pickle.dump(inputTree,fw) #序列化对象，保存到磁盘
    print("store %s successfully!"%(filename))
    fw.close()
    
def getTree(filename):
    fr = open(filename,'rb') 
    return pickle.load(fr) #从磁盘读取,从file中读取一个字符串，并将它重构为原来的python对象

filename = u"D:\gitcode\Machine-learning-in-action\decisionTree\myTree.txt"
storeTree(myTree,filename)
tree = getTree(filename)
print(tree)

