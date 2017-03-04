#coding=utf-8
from numpy import *

def loadDataSet(fileName,delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float,line) for line in stringArr]
    return mat(datArr)

def pca(dataMat,topNfeat=99999):
    meanVals = mean(dataMat)
    meanRemoved = dataMat-meanVals
    covMat = cov(meanRemoved,rowvar=0)
    eigVals,eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)
    eigValInd = eigValInd[:-(topNfeat+1):-1] #逆序
    redEigVects = eigVects[:,eigValInd] # n*k
    print 'red',shape(redEigVects)
    lowDDataMat = meanRemoved * redEigVects # m*n n*k
    reconMat = (lowDDataMat*redEigVects.T)+meanVals # m*k k*n
    print 'lowD',shape(lowDDataMat)
    return lowDDataMat,reconMat

def replaceNanWithMean():
    datMat = loadDataSet('secom.data',' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i])
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal
    return datMat

if __name__=='__main__':
    dataMat = loadDataSet('testSet.txt')
    lowDat,reconMat = pca(dataMat,1)
    # print shape(lowDat)

    # datMat = replaceNanWithMean()
    # meanVals = mean(datMat)
    # meanRemoved = datMat-meanVals
    # covMat = cov(meanRemoved,rowvar=0)
    # eigVals,eigVects = linalg.eig(mat(covMat))
    # print eigVals




