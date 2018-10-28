
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*- #可以使用中文注释
#朴素贝叶斯用于文本分类

#以网站发帖留言，作为文本数据集合 
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

#根据所有的留言，构建词汇表（包含所有的单词token）
def createVocabularyList(dataSet):
    vocabularySet = set([])  #create empty set，利用set去重的特点
    for document in dataSet:
        vocabularySet = vocabularySet | set(document) #union of the two sets ,取或操作(并集)
    return list(vocabularySet)

#词序模型，将某个输入留言inputSet转化为关于词汇表的0/1向量
def setOfWordsVec(vocabularyList, inputSet):
    returnVec = [0]*len(vocabularyList)
    for word in inputSet:
        if word in vocabularyList:
            returnVec[vocabularyList.index(word)] = 1 #留言中的某个单词在词汇表中，则为1，否则为0
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec


#现在测试一下效果
postingList,classVec = loadDataSet()
vocabularyList = createVocabularyList(postingList)
print("the vocabulary list is:\n",vocabularyList)
returnVec = setOfWordsVec(vocabularyList,postingList[0])
print("post0 vector=\n",returnVec)

#现在将所有的留言，都转化为0/1词汇表特征向量，作为trainVec
trainVec = []
for post in postingList:
    trainVec.append(setOfWordsVec(vocabularyList,post))

print("all post vector are:\n",trainVec)


# In[2]:


import numpy as np

#其实bayes的核心函数就：NaiveBayes0 + classify

#bayes求得，特征向量的概率，p(w/y)
def NaiveBayes0(trainMatrix,trainCategory):
    numTrain = len(trainMatrix) #训练的文档个数
    numWords = len(trainMatrix[0]) #词汇表的大小，即特征的大小
    p1 = sum(trainCategory)/float(numTrain)  #分类1的概率p(y=1)，这里是二分类，所以，p0=1-p1
    p0Num = np.zeros(numWords); p1Num = np.zeros(numWords)      #初始化为0
    p0Denom = 0.0; p1Denom = 0.0                        
    print('this is trainMatrix1')
    for i in range(numTrain):
        if trainCategory[i] == 1:
            print(trainMatrix[i])
            p1Num += trainMatrix[i] #y=1条件下，统计某个单词出现的个数，用于计算p(w/y=1)
            p1Denom += sum(trainMatrix[i]) #累计y=1的所有单词数量
        else:
            p0Num += trainMatrix[i] #y=0条件下，统计某个单词出现的个数，用于计算p(w/y=0)
            p0Denom += sum(trainMatrix[i]) #累计y=0的所有单词数量
    p1Vect = p1Num/float(p1Denom)     #p(w/y=1)   
    p0Vect = p0Num/float(p0Denom)     #p(w/y=0)  
    return p0Vect,p1Vect,p1



#this is function by jackson,ugly but work well
def NaiveBayes(featureMatrix,classMatrix):
    featureMatrix = np.array(featureMatrix)
    classMatrix = np.array(classMatrix)
    line = len(classMatrix)
    pclass1 = sum(classMatrix[classMatrix == 1])/float(line)
    class1matrix = featureMatrix[classMatrix == 1]
    class0matrix = featureMatrix[classMatrix == 0]
    pclass1_vec = []
    pclass0_vec = []
    for i in range(len(class1matrix[0])):
        pclass1_vec.append(sum(class1matrix[:,i]))
    for j in range(len(class0matrix[0])):
        pclass0_vec.append(sum(class0matrix[:,j]))
    under1 = sum(pclass1_vec)
    pclass1_vec = [x/float(under1) for x in pclass1_vec]
    under0 = sum(pclass0_vec)
    pclass0_vec = [x/float(under0) for x in pclass0_vec]
    return pclass0_vec,pclass1_vec,pclass1


#测试一下
p0Vect,p1Vect,p1 = NaiveBayes(trainVec,classVec)
print('=======')
print(p0Vect)
print(p1Vect)
print(p1)
print('======')
print("p(y=1) = ",p1 ) #classVec = [0,1,0,1,0,1] ,so p1=0.5
print("y=1,the word with the max Probability=(%f), is(%s)"      %(np.max(p1Vect),vocabularyList[np.argmax(p1Vect)]))


# In[3]:


#根据李航的《统计学习方法》，上面的估计概率的 方法叫做————“极大似然估计”
#有两个缺点：
# 1，有些概率=0，那么，后面的概率相乘，会影响计算结果，所以改进-----》李航书上：贝叶斯估计概率
# 2,很多小数相乘，会下溢出，所以加上log

#修改一下上面的函数

#bayes求得，特征向量的概率，p(w/y)
def NaiveBayes1(trainMatrix,trainCategory):
    numTrain = len(trainMatrix) #训练的文档个数
    numWords = len(trainMatrix[0]) #词汇表的大小，即特征的大小
    p1 = sum(trainCategory)/float(numTrain)  #分类1的概率p(y=1)，这里是二分类，所以，p0=1-p1
    p0Num = np.ones(numWords); p1Num = np.ones(numWords)      #change to 1 ,一般sigma = 1
    p0Denom = 2.0; p1Denom = 2.0                        # change to 2.0 因为有两个类别
    for i in range(numTrain):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i] #y=1条件下，统计某个单词出现的个数，用于计算p(w/y=1)
            p1Denom += sum(trainMatrix[i]) #累计y=1的所有单词数量
        else:
            p0Num += trainMatrix[i] #y=0条件下，统计某个单词出现的个数，用于计算p(w/y=0)
            p0Denom += sum(trainMatrix[i]) #累计y=0的所有单词数量
    p1Vect = np.log(p1Num/float(p1Denom))         #change to log()
    p0Vect = np.log(p0Num/float(p0Denom))          #change to log()
    return p0Vect,p1Vect,p1

#分类的函数，计算输入为x的条件下，属于每个类别的概率
def classify(testVec, p0Vec, p1Vec, pClass1):
    p1 = sum(testVec * p1Vec) + np.log(pClass1)    #因为是log,所以这里是求和以及+号操作
    #p(y=1/w) = p(w/y=1) * p(y=1),注意这里需要乘上，testVec,过滤掉那些为0的特征的概率
    p0 = sum(testVec * p0Vec) + np.log(1.0 - pClass1) #p(y=0) = 1 - p(y=1)
    if p1 > p0: #选取最大的概率的类  
        return 1
    else: 
        return 0
    


# In[4]:


#测试一下，看看对于一段留言的分类效果
p0V,p1V,p1 = NaiveBayes1(trainVec,classVec)
test0 = ['love', 'my', 'dalmation']
testVec0 = setOfWordsVec(vocabularyList, test0)
print(test0,'classified as: ',classify(testVec0,p0V,p1V,p1))
test1 = ['stupid', 'garbage']
testVec1 = setOfWordsVec(vocabularyList, test1)
print(test1,'classified as: ',classify(testVec1,p0V,p1V,p1))


# In[5]:


# #下面我们将贝叶斯模型用于垃圾邮件分类


# #词袋模型，需要统计某个token出现的次数
# def bagOfWordsVec(vocabularyList, inputSet):
#     returnVec = [0]*len(vocabularyList)
#     for word in inputSet:
#         if word in vocabularyList:
#             returnVec[vocabularyList.index(word)] += 1 #出现一次，累加一次
#     return returnVec


# # In[6]:


# import re
# #文档处理，
# def textParse(bigString):    #input is big string, #output is word list
#     listOfTokens = re.split(r'\W*', bigString) #只需要字符和数字
#     return [tok.lower() for tok in listOfTokens if len(tok) > 2] #变成小写，过滤长度小于3的字符串 
    
# def spamTest():
#     docList=[]; classList = []; fullText =[]
#     for i in range(1,26): #每个文件夹，有25个文件
#         wordList = textParse(open('bayes/email/spam/%d.txt' % i).read())
#         docList.append(wordList)
#         fullText.extend(wordList)
#         classList.append(1)
#         wordList = textParse(open('bayes/email/ham/%d.txt' % i).read())
#         docList.append(wordList)
#         fullText.extend(wordList)
#         classList.append(0)
#     vocabularyList = createVocabularyList(docList)#create vocabulary
#     trainingSet = range(50); testSet=[]   #create test set,这里其实是val set，而且只保存了下标
#     for i in range(10):
#         randIndex = int(np.random.uniform(0,len(trainingSet)))
#         testSet.append(trainingSet[randIndex])
#         del(trainingSet[randIndex])  #随机选取10个，并且从train中删除
#     trainMat=[]; trainClasses = []
#     for docIndex in trainingSet:#train the classifier (get probs) trainNB0
#         trainMat.append(bagOfWordsVec(vocabularyList, docList[docIndex]))
#         trainClasses.append(classList[docIndex])
#     p0V,p1V,pSpam = NaiveBayes1(trainMat,trainClasses) #学习得到的train的概率
#     errorCount = 0
#     for docIndex in testSet:        #classify the remaining items
#         wordVector = bagOfWordsVec(vocabularyList, docList[docIndex])
#         if classify(wordVector,p0V,p1V,pSpam) != classList[docIndex]: #用于test/val
#             errorCount += 1
#             print "classification error",docList[docIndex]
#     print 'the error rate is: ',float(errorCount)/len(testSet)


# # In[7]:


# #因为是随机选取10个留存作为val set,所以，error rate 会变化
# spamTest()


# # In[8]:


# spamTest()

