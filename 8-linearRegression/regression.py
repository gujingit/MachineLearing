#coding=utf-8
from numpy import *
import matplotlib.pyplot as plt


def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))-1
    dataMat =[];labelMat=[]
    fr = open(fileName)
    for line in fr.readlines():
        lineArr=[]
        currLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(currLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(currLine[-1]))
    return dataMat,labelMat


def standRegress(xArr,yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0.0:
        print 'This matrix is singular,cannot do inverse'
        return
    ws = xTx.I*(xMat.T*yMat)
    return ws


def drawData(xArr,yArr,ws):
    xMat = mat(xArr);yMat = mat(yArr)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # print 'null',xMat[:,1]
    # print 'flatten',xMat[:,1].flatten()
    ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten.A[0])
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:,1],yHat)
    plt.show()


def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr);yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        weights[j,j]=exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T*(weights*xMat)
    if linalg.det(xTx) ==0.0:
        print 'The matrix cannot do inverse'
        return
    ws = xTx.I*(xMat.T*(weights*yMat))
    return testPoint * ws


def lwlrTest(testArr,xArr,yArr,k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat


def rssError(yArr,yHatArr):
    return ((yArr-yHatArr)**2).sum()


#岭回归，用于n>m时
def ridgeRegress(xMat,yMat,lam=0.2):
    xTx = xMat.T*xMat
    denom = xTx + eye(shape(xMat)[1])*lam
    if linalg.det(denom) == 0:
        print 'This matrix cannot do inverse'
        return
    ws = denom.I * (xMat.T*yMat)
    return ws

def ridgeTest(xArr,yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    yMean = mean(yArr,0)
    yMat = yMat-yMean
    xMeans = mean(xMat,0)
    xVar = var(xMat,0)
    xMat = (xMat-xMeans)/xVar
    numTestPts =30
    wMat = zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegress(xMat,yMat,exp(i-10))
        wMat[i,:] = ws.T
    return wMat


def regularize(xMat):
    inMat = xMat.copy()
    xMeans = mean(inMat,0)
    xVal = var(inMat,0)
    inMat = (inMat-xMeans)/xVal
    return inMat


def stageWise(xArr,yArr,eps=0.01,numIt=100):
    xMat =  mat(xArr);yMat = mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat -yMean # y 除以方差后，系数会变小；除不除都可以
    xMat = regularize(xMat)
    m,n = shape(xMat)
    returnMat = zeros((numIt,n))
    ws = zeros((n,1));wsTest = ws.copy();wsMax = ws.copy()
    for i in range(numIt):
        print ws.T
        lowestError = inf;
        for j in range(n):
            for sign in [-1,1]:
                wsTest =ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat*wsTest
                rssE = rssError(yMat.A,yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:] = ws.T
    return returnMat


def crossValidation(xArr,yArr,numVal=10):
    m = len(yArr)
    indexList = range(m)
    errorMat = zeros((numVal,30))
    for i in range(numVal):
        trainX=[];trainY=[];testX=[];testY=[]
        random.shuffle(indexList)
        for j in range(m):
            if j < m*0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX,trainY)
        for k in range(30):
            matTestX = mat(testX);matTrainX = mat(trainX)
            meanTrain = mean(matTrainX,0)
            varTrain = var(matTrainX,0)
            matTestX = (matTestX-meanTrain)/varTrain
            yEst = matTestX*mat(wMat[k,:]).T+mean(trainY)
            errorMat[i,k]=rssError(yEst.T.A,array(testY))
    meanError = mean(errorMat,0)
    minError = float(min(meanError))
    bestWeights = wMat[nonzero(meanError==minError)]
    xMat = mat(xArr);yMat = mat(yArr).T
    meanX = mean(xMat,0);varX = var(xMat,0)
    unReg = bestWeights/varX #因为之前y没有除
    print "the best model from Ridge Regress is:\n",unReg
    print "with constant term: ", -1 * sum(multiply(meanX, unReg)) + mean(yMat)



if __name__=="__main__":
    ## 1
    # dataMat,labelMat = loadDataSet('ex0.txt')
    # ws = standRegress(dataMat,labelMat)
    # xMat = mat(dataMat)
    # yMat = mat(labelMat)
    # #drawData(dataMat,labelMat,ws)
    # yHat = xMat * ws
    # print yHat.T.shape,' ',yMat.shape
    # print corrcoef(yHat.T,yMat) #计算相关系数

    ## 2
    # xArr,yArr = loadDataSet('ex0.txt')
    # yHat = lwlrTest(xArr,xArr,yArr,0.003)
    # xMat = mat(xArr)
    # srtInd = xMat[:,1].argsort(0)
    # xSort = xMat[srtInd][:,0,:]
    # print yHat.shape
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(xSort[:,1],yHat[srtInd])
    # ax.scatter(xMat[:,1].flatten().A[0],mat(yArr).T.flatten().A[0],c='red')
    # plt.show()

    ## 3
    # abX,abY = loadDataSet('abalone.txt')
    # ridgeWeights = ridgeTest(abX,abY)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(ridgeWeights)
    # plt.show()

    ## 4
    xArr,yArr = loadDataSet('abalone.txt')
    crossValidation(xArr,yArr)

    dataMat, labelMat = loadDataSet('abalone.txt')
    ws = standRegress(dataMat,labelMat)
    xMat = mat(dataMat)
    yMat = mat(labelMat)
    meanX = mean(xMat,0);varX = var(xMat,0)
    print "the best model from Stand Regress is:\n", ws
    print "with constant term: ", -1 * sum(multiply(meanX, ws)) + mean(yMat)




