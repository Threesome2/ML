# knn2.py
import numpy as np 
from matplotlib import pyplot as plt

# 读取数据
def data_read(filename):
    fr = open(filename)
    data_lines = fr.readlines()
    a = np.shape(data_lines)[0]
    data_features = np.zeros([a,3])
    data_labels = []
    for index,line in zip(range(a),data_lines):
        line = line.strip()
        data_split = line.split('\t')
        data_features[index,:] = data_split[0:-1]
        data_labels.append(data_split[-1])
    # print('========')
    # print(np.shape(data_features))
    return(data_features,data_labels)


# 归一化
def data_Norm(fr):
    col = np.shape(fr)[1]
    for k in range(col):
        fr_min = fr[:,k].min()
        # print(fr_min)
        fr_max = fr[:,k].max()
        fr[:,k] = [(x - fr_min)/(fr_max - fr_min) for x in fr[:,k]]
    return fr,fr_min,fr_max

# 绘图
def data_plot(data_features,labels):
    labels=np.array(labels)
    plt.subplot(2,2,1)
    for t,marker,c in zip(range(3),'xo>','rgy'):
        plt.scatter(data_features[labels==str(t+1),0],data_features[labels==str(t+1),1],marker=marker,c=c)
    plt.subplot(2,2,2)
    for t,marker,c in zip(range(3),'xo>','rgy'):
        plt.scatter(data_features[labels==str(t+1),0],data_features[labels==str(t+1),2],marker=marker,c=c)
    plt.subplot(2,2,3)
    for t,marker,c in zip(range(3),'xo>','rgy'):
        plt.scatter(data_features[labels==str(t+1),1],data_features[labels==str(t+1),2],marker=marker,c=c)
    plt.show()



# # 标准化
# def data_standardized():
#     return
 

# # 二值化(将数值特征向量转换为布尔类型向量)
# def data_binarized():


# # 独热编码
# def OneHotEncoding():
#     return

# knn算法
def knn_cal(inX,data_features,data_labels,k):
    # 计算距离
    sqinX_sum = distance_Euclid(inX, data_features)
    # 距离排序
    out_argsorted = sqinX_sum.argsort()
    # k个投票结果放到字典里
    dict_label = {}
    for i in range(k):
        label = data_labels[out_argsorted[i]]
        if label in dict_label:
            dict_label[label] += 1
        else:
            dict_label[label] = 1
    print(dict_label)
    # 找出最大的那个label返回
    label_max = ''
    label_compare = 0
    for item in dict_label:
        if dict_label[item] > label_compare:
            label_max = item
            label_compare = dict_label[item]
    return label_max



 # 欧氏距离
def distance_Euclid(inX,data_features):
    inX_tile = np.tile(inX,(np.shape(data_features)[0],1))
    inX_minus = inX_tile - data_features
    sqinX = inX_minus**2
    sqinX_sum = np.sum(sqinX,axis = 1)**0.5
    return sqinX_sum



# # 曼哈顿距离
# def distance_Manhattan():


#  # 切比雪夫距离
# def distance_Chebyshev():


# # 闵可夫斯基距离
# def distance_Minkowski():


# # 标准化欧氏距离
# def distance_Standardized_Euclidean():


# # 马氏距离
# def distance_Mahalanobis():


def test(filename):
    data_features,data_labels = data_read(filename)
    data_features,data_min,data_max = data_Norm(data_features)
    auc = 0
    for i in range(int(np.shape(data_labels)[0]*0.1)):
        # 测试数据是按行逢九进一，也可以用别的规则，从原数据提取
        inX = data_features[i*9 + 1,:]
        prediction = knn_cal(inX, data_features, data_labels, 5)
        if prediction == data_labels[i*9+1]:
            auc += 1
    auc = auc/(np.shape(data_labels)[0]*0.1)
    print('精确度是：%s'%auc)


def main():
    filename = u'D:\gitcode\Machine-learning-in-action\k-Nearest Neighbor\datingTestSet2.txt'
    test(filename)

# main()


def dict_test(dicts):
    print(dicts)
    label={}
    for item in dicts:
        print(item)
        if item not in label.keys():
            label[item] = 0
        label[item] += 1
    print(label)



dicts=['a','a','a','a','b','b','c',]
dict_test(dicts)
