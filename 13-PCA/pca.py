#coding=utf-8
from numpy import *

def loadDataSet(fileName,delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float,line) for line in stringArr]
    return mat(datArr)

def pca(dataMat,topNfeat=99999):
    meanVals = mean(dataMat)
    print 'data',shape(dataMat)
    meanRemoved = dataMat-meanVals
    covMat = cov(meanRemoved,rowvar=0)
    eigVals,eigVects = linalg.eig(mat(covMat))
     # print 'shape', shape(eigVects) eigVects n*n
    eigValInd = argsort(eigVals)
    eigValInd = eigValInd[:-(topNfeat+1):-1] #逆序
    redEigVects = eigVects[:,eigValInd] # n*k
    lowDDataMat = meanRemoved * redEigVects # PCA 降维 A = V*diag(lambda)*V.T A*V = V*diag(lambda)*V.T*V = V*diag(lambda) (n*k)
    reconMat = (lowDDataMat*redEigVects.T)+meanVals # 恢复高维数据 redEigVects为正交矩阵,所以矩阵转置等于矩阵逆
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




