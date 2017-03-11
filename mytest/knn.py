#coding=utf-8
import numpy as np
from sklearn import neighbors
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt

def loadDataSet(filename):
    strArr = [line.strip().split('\t') for line in open(filename).readlines()]
    dataSet = [map(float,line) for line in strArr]
    dataMat = np.mat(dataSet)
    m,n = np.shape(dataMat)
    return dataMat[:,:n-1],dataMat[:,-1]


if __name__=="__main__":
    x,y = loadDataSet('../2-knn/datingTestSet2.txt')

    #拆分数据集
    m = np.shape(x)[0]
    kf = KFold(m,n_folds=5,shuffle=True) #1000分为5份
    clf = neighbors.KNeighborsClassifier(n_neighbors=3)
    for iteration,data in enumerate(kf,start=1):
        clf.fit(x[data[0]],np.ravel(y[data[0]]))
        answer = clf.predict(x[data[1]])
        print 'iteration',iteration
        print(classification_report(y[data[1]],answer))

    #训练KNN分类器
    # x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
    # clf = neighbors.KNeighborsClassifier(n_neighbors=3)
    # clf.fit(x_train,np.ravel(y_train))
    # answer = clf.predict(x_test)
    # print(classification_report(y_test,answer))

    # precision,recall,thresholds = precision_recall_curve(y_test,answer) 二分类问题
    # metric = cross_val_score(clf,x,y,cv=5,scoring='accuracy').mean() 二分类问题