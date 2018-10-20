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
    print('len(vocabularySet)')
    print(len(vocabularySet))
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
print('postingList is')
print(postingList)
vocabularyList = createVocabularyList(postingList)
print("the vocabulary list is:\n",vocabularyList)
returnVec = setOfWordsVec(vocabularyList,postingList[0])
print("post0 vector=\n",returnVec)

#现在将所有的留言，都转化为0/1词汇表特征向量，作为trainVec
trainVec = []
for post in postingList:
    trainVec.append(setOfWordsVec(vocabularyList,post))

print("all post vector are:\n",trainVec)