from numpy import *
import operator


def createDataSet():
    group = array([1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


"""
输入向量inX，训练样本集dataSet，标签向量labels，选择最近邻居的数目k
labels元素数==dataSet行数
"""
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=true)
    return sortedClassCount[0][0]
