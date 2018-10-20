def NaiveBayes(featureMatrix,classMatrix):
    line = len(classMatrix)
    pclass1 = sum(classMatrix[classMatrix == 1])/float(line)
    class1matrix = featureMatrix[classMatrix == 1]
    class0matrix = featureMatrix[classMatrix == 0]
    pclass1_vec = []
    pclass0_vec = []
    for i in range(len(class1matrix[0])):
        pclass1_vec.append(sum(class1matrix[:,i])/float(line))
    for i in range(len(class0matrix[0])):
        pclass0_vec.append(sum(class1matrix[:,i]))/float(line)
    return pclass1,pclass0_vec,pclass1_vec





