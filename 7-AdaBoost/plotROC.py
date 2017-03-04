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
