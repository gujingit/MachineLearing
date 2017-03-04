#coding=utf-8
from numpy import *
import matplotlib.pyplot as plt


def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(lineArr[2])
    return dataMat,labelMat


def sigmoid(inX):
    #return longfloat(1.0/(1+exp(-inX))) #存在计算溢出问题
    return tanh(inX)


def gradAscent(dataMatrix,classLabels,numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)

    for j in range(numIter):
        dataIndex = range(m) # 随机抽样的范围 某种随机抽样方法（放回抽样）
        for i in range(m):
            alpha = 4/(1.0+i+j)+0.01
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = int(classLabels[randIndex]) - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights


def plotBestFit(weights):
    dataMat,labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1=[]
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x = arange(-3.0,3.0,0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def classifyVector(inX,weights):
    prob = sigmoid(sum(inX*weights))
    if prob >0.5: return 1.0
    else: return 0


def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet =[];trainingLabels=[]
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = gradAscent(array(trainingSet),trainingLabels,500)
    errorCount =0.0;numTextVec=0.0
    for line in frTest.readlines():
        numTextVec += 1
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if classifyVector(lineArr,trainWeights) != float(currLine[21]):
            errorCount += 1.0
    errorRate = errorCount / numTextVec
    print 'the error rate of this test is: %f' % errorRate
    return errorRate


def multiTest():
    numTests = 10
    errorSum =0.0
    for k in range(numTests):
        errorSum += colicTest()
    print 'after %d iterations the average error rate is: %f' % (numTests, errorSum /float(numTests))

if __name__=='__main__':
    multiTest()