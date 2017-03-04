#coding=utf-8
import numpy as np

def loadDataSet(filename):
    stringData = [line.strip().split(',') for line in open(filename).readlines()]
    dataSet = [map(float,line) for line in stringData]
    return np.mat(dataSet)

if __name__=="__main__":
    dataSet = loadDataSet('pima-indians-diabetes.data')
    print dataSet.shape
    print dataSet[1,:]
