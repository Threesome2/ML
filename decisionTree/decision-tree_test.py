# decision-tree_test.py
import math
import numpy as np

def create_data():
    datasets=np.array(
    [[1,1,'yes'],
    [1,1,'yes'],
    [1,0,'no'],
    [0,1,'no'],
    [0,1,'no']])
    features = datasets[:,:-1]
    featureNames = ['no surfacing','flippers'] #不浮出水面是否存活 ，有无脚蹼
    print('featureNames:',featureNames)
    labels = datasets[:,-1]
    return datasets,featureNames

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
    this fuction is used to choose the best feature
    for zhe decitionTree
    '''
    best_infoGain = 0.0 
    base_Entropy = Entropy(datasets)
    for i in range(np.shape(datasets)[1]-1):
        col_value_data = set(datasets[:,i])
        new_entropy = 0.0
        for value in col_value_data:
            new_data = datasets[datasets[:,i]==str(value),-1]
            prop = len(new_data)/float(len(datasets))
            new_entropy += prop*Entropy(new_data)
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



def createTree(datasets,featureNames):
    '''
    1.choose best_feature_name to construct myTree
    2.drop the best feature and construct new_data 
    with the left features to replace the original datasets.
    3.call this function itself to construct next level of myTree
    4.set the  conditions to get off this function
        conditon1:only one kind of label left in datasets
        conditone2:no more features left in datasets

    '''
    classList = datasets[:,-1]
    if len(datasets[0]) == 1:
        # print('no features provided')
        return majorityCnt(classList)
    # print('classList',classList)
    num = np.sum(classList == classList[0])
    if num == len(classList): #如果所有的类都一样，第0类的个数==长度
        return classList[0]
    best_feature = chooseBestFeatureToSplit(datasets)
    myTree={}
    best_feature_name = featureNames[best_feature]
    myTree[best_feature_name] = {}
    # print('this is myTree:')
    # print(myTree)
    featureNames.pop(best_feature)
    featureValue = datasets[:,best_feature]
    uniqueValues = set(featureValue)
    # print(featureNames)
    for value in uniqueValues:
        subNames = featureNames[:]
        myTree[best_feature_name][value] = createTree(splitDataset(datasets, best_feature, value),subNames)
    return myTree


def test_func():
    '''
    to test the majorityCnt functin
    '''
    #测试
    c = [1,0,0,0,0,1,1,1,0,0,2,2,2,2]
    print("the majorityClassCount is 0\n",majorityCnt(c))










# 预测函数
def feature_calssify(tree_dict,featureNames,testVec):
    if isinstance(tree_dict,dict):
        for key in tree_dict:
            if isinstance(tree_dict[key],dict):
                for value in tree_dict[key]:
                    if str(testVec[key == featureNames]) == value:
                        if not isinstance(tree_dict[key][value],dict):
                            # 非字典结构直接返回结果
                            return tree_dict[key][value]
                        else:
                            tree_dict = tree_dict[key][value]
                            # 字典结构回调
                            return feature_calssify(tree_dict,featureNames,testVec)
            else:
                # 非字典结构直接返回结果
                return tree_dict[key]
            



# ...






def main():
    datasets,featureNames = create_data()
    entropy = Entropy(datasets[:,2])
    data_splited_value = splitDataset(datasets,0,1)
    best_feature = chooseBestFeatureToSplit(datasets)
    myTree = {}
    myTree = createTree(datasets,featureNames)

    pred=feature_calssify(myTree, featureNames, [1,1])
    print(pred)


main()


