from numpy import *
import operator


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


"""
输入向量inX，训练样本集dataSet，标签向量labels，选择最近邻居的数目k
labels元素数==dataSet行数
"""
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # 读取dataSet第一维度的长度→行数
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # 距离公式里对应项相减
    sqDiffMat = diffMat ** 2  # 平方
    sqDistances = sqDiffMat.sum(axis=1)  # 平方项相加
    distances = sqDistances ** 0.5  # 开方
    sortedDistIndicies = distances.argsort()  # 将distances中的元素从小到大排列，提取其对应的index(索引)，然后输出到sDI
    classCount = {}  # 定义字典
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]  # 提取sDI中值对应的标签
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # 以标签为key计算次数
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # 将字典中每个key对应的次数从大到小排序，items返回字典中的每项k = (key：value)组成的无序列表，operator.itemgetter(1)表示按照k(1)位置的值排序，即按照value排序
    return sortedClassCount[0][0]  # 返回第一个元组中的第一个值，即key，即标签


def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)  # 获取文件行数
    returnMat = zeros((numberOfLines,3))  # 设为3只是为了简化处理，可修改 **并没有中括号的写法
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]  # 按行赋值 将lFL的前三个元素赋给returnMat的第index行
        classLabelVector.append(int(listFromLine[-1]))  # 把每次lFL的最后一个元素加入cLV **不设int的话会作为字符串处理
        index += 1
    return returnMat,classLabelVector
