#coding=utf-8
from numpy import *
import matplotlib.pyplot as plt

def loadSimpData():
    dataMat = matrix([[1.0,2.1],
                      [2.0,1.1],
                      [1.3,1.0],
                      [1.0,1.0],
                      [2.0,1.0]])
    classLabels =[1.0,1.0,-1.0,-1.0,-1.0]
    return dataMat,classLabels

def drawData():
    dataMat,classLabels = loadSimpData()
    xcord0=[];ycord0=[]
    xcord1=[];ycord1=[]
    for i in range(len(classLabels)):
        if classLabels[i]==1.0:
            print dataMat[0,0]
            xcord0.append(dataMat[i,0])
            ycord0.append(dataMat[i,1])
        else:
            xcord1.append(dataMat[i,0])
            ycord1.append(dataMat[i,1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord0,ycord0,marker='s',s=90)
    ax.scatter(xcord1,ycord1,marker='o',s=50,c='red')
    plt.title('decision tree stump test')
    plt.show()


def stumpClassify(dataMatrix,dimen,threshVal,threshIneq): # dimen表示第dimen个特征
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] =-1.0

    return retArray

# 单层决策树 弱分类器
def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0;bestStump={};bestClassEst = mat(zeros((m,1)))
    minError = inf
    for i in range(n):
        rangeMin = dataMatrix[:,i].min()
        rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMax - rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt','gt']:
                threshVal = (rangeMin+float(j)*stepSize)
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr = mat(ones((m,1)))
                errArr[predictedVals==labelMat]=0
                weightedError = D.T*errArr
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClassEst


def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)
        #print "D:",D.T
        alpha = float(0.5*log((1-error)/max(error,1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        #print 'classEst: ',classEst.T
        expon = multiply(-1*alpha*mat(classLabels).T,classEst)
        D = multiply(D,exp(expon))
        D = D/D.sum()
        aggClassEst += alpha*classEst
        #print 'aggClassEst: ', aggClassEst.T
        aggErrors = multiply(sign(aggClassEst)!=
                             mat(classLabels).T,ones((m,1)))
        errorRate = aggErrors.sum()/m
        print 'total error: ',errorRate,'\n'
        if errorRate == 0: break
    return weakClassArr,aggClassEst


def adaClassify(dataToClass,classifierArr):
    dataMatrix = mat(dataToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],
                                 classifierArr[i]['thresh'],
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
        print aggClassEst
    return sign(aggClassEst)


def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat =[];labelMat =[]
    fr = open(fileName)
    for line in fr.readlines():
        lineArr=[]
        currArr = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(currArr[i]))
        dataMat.append(lineArr)
        labelMat.append(float(currArr[-1]))
    return dataMat,labelMat


#coding=utf-8
from numpy import *
import matplotlib.pyplot as plt


def plotROC(predictLabels,classLabels):
    cur = (1.0,1.0)
    ySum = 0.0
    numPosClas = sum(array(classLabels)==1.0)
    yStep = 1/float(numPosClas)
    xStep = 1 / float(len(classLabels)-numPosClas)
    sortedIndicies = predictLabels.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111) # or ax = fig.add_subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index]==1.0:
            delX = 0; delY = yStep
        else:
            delX = xStep
            delY =0
            ySum += cur[1]
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY],c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    ax.axis([0,1,0,1])
    plt.show()
    print 'the Area under the curve is : ',ySum * xStep


if __name__=="__main__":
    dataMat,labelMat = loadDataSet('horseColicTraining2.txt')
    classifierArr,aggClassEst = adaBoostTrainDS(dataMat,labelMat,10)
    plotROC(aggClassEst.T,labelMat)

    # dataMat,labelMat = loadDataSet('horseColicTest2.txt')
    # predict = adaClassify(dataMat,classifierArr)
    # errArr = mat(ones((67,1)))
    # errorRate = errArr[predict!=mat(labelMat).T].sum()/67
    # print 'errorRate: ',errorRate














