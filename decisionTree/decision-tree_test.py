# decision-tree_test.py
import math
import numpy as np

def create_data():
    datasets=np.array(
    [[1,1,'maybe'],
    [1,1,'yes'],
    [1,0,'no'],
    [0,1,'no'],
    [0,1,'no']])
    features = datasets[:,:-1]
    labels = datasets[:,-1]
    return datasets,features

def digitalize_Data(datasets):
    datasets[datasets[:,2] == 'yes',2] = 0
    datasets[datasets[:,2] == 'no',2] = 1
    datasets[datasets[:,2] == 'maybe',2] = 2
    return datasets


# 统计标签和数量并放到字典里
def Entropy(datasets):
    num = np.shape(datasets)[0]
    # num = len(datasets)
    dict_labels = {}
    # 把标签和数量存到dict_labels字典里
    for featVec in datasets:
        currentLabel = featVec[-1]
        if currentLabel not in dict_labels:
            dict_labels[currentLabel] = 0
        dict_labels[currentLabel] += 1
    entropy = 0.0
    # 汇总计算熵
    for item in dict_labels:
        prop = float(dict_labels[item])/num
        entropy += -prop*math.log(prop,2)
    return entropy

def splitDataset(datasets,axis,value):
    data_splited = np.delete(datasets[:],axis,axis=1)
    data_splited_value = data_splited[datasets[:,axis]==str(value),:]
    return data_splited_value

def chooseBestFeatureToSplit(datasets):
    '''
    this is a description for this function.
    this fuction is used for choose the best feature
    for zhe decitionTree
    '''
    best_infoGain = 0.0 
    base_Entropy = Entropy(datasets)
    for i in range(np.shape(datasets)[1]-1):
        col_value_data = set(datasets[:,i])
        new_entropy = 0.0
        for value in col_value_data:
            print(value,col_value_data)
            new_data = datasets[datasets[:,i]==str(value),2]
            print(new_data)
            prop = len(new_data)/float(len(datasets))
            new_entropy += prop*Entropy(new_data)
        print('new_entropy',new_entropy)
        info_gain = base_Entropy - new_entropy
        if info_gain > best_infoGain:
            best_infoGain = info_gain
            best_feature = i
    return best_feature
    
def majorityCnt(classList):
    '''
    to sort dict and return the max item
    
    '''
    classcount={}
    for item in classList:
        if item not in classcount:
            classcount[item] = 0 
        classcount[item] += 1
    max_item_value = 0 
    for item in classcount:
        if classcount[item]>max_item_value:
            max_label = item 
            max_item_value = classcount[item]
    return max_label


def test_func():
    '''
    to test the majorityCnt functin
    '''
    #测试
    c = [1,0,0,0,0,1,1,1,0,0,2,2,2,2]
    print("the majorityClassCount is 0\n",majorityCnt(c))





def main():
    datasets,features = create_data()
    print(datasets[:,2])
    entropy = Entropy(datasets[:,2])
    print(entropy)
    # featureList = [example[0] for example in datasets]
    # print(featureList)
    datasets= digitalize_Data(datasets)
    print(datasets)
    data_splited_value = splitDataset(datasets,0,1)
    print(data_splited_value)
    best_feature = chooseBestFeatureToSplit(datasets)
    print('best_feature is :',best_feature)
# main()


# test_func()