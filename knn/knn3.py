# knn3.py

filename=(u"D:\gitcode\Machine-learning-in-action\k-Nearest Neighbor\datingTestSet2.txt")       #load data setfrom file


import numpy as np 




def data_read(filename):
    fr = open(filename,'r')
    data_features = []
    data_labels = []
    while True: 
        # print(fr.readline())
        a=fr.readline()
        a=a.strip()
        if a != '':
            data_split = a.split('\t')
            data_features.append(data_split[0:-1])
            data_labels.append(data_split[-1])
        else:
            break

    return data_features,data_labels


def Norm(data_features):
    # col = np.shape(data_features)[1]
    # print('col:',col)
    # print(data_features[:,0])
    # for i in range(col):
    #     data_features_min = min(list(data_features[:,i]))
    #     print(data_features_min)
    #     data_features_max = max(list(data_features[:,i]))
    #     print('======')
    #     print(data_features_max)
    #     data_range = data_features_max - data_features_min
    #     data_features[:,i] = [(x - data_features_min)/data_range for x in data_features[:,i]]
    # data_features=np.ones((1000,3))  
    print(data_features) 
    print(np.shape(data_features))
    a = np.min(data_features,0)
    print(a)
    return data_features



def main():
    data_features,data_labels = data_read(filename)
    # print(fr)
    data_features = np.array(data_features).astype(float)
    print(np.shape(data_features))
    print(data_features)
    # print(np.shape(data_labels))
    # print(data_features[0:5,:])
    data_features = Norm(data_features)
    # print(data_features[0:5,:])




main()

# def t1(num):
#     num = num + num
#     print(num)

# def test_python():
#     a=[10]
#     t1(a)
#     print(a)

# test_python()
