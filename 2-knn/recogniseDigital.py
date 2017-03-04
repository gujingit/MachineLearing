from numpy import *
from os import listdir
import operator

def classify0(inX,dataSet,labels,k):
    m = dataSet.shape[0]
    diffMat = dataSet - tile(inX, (m, 1))
    diffMat = diffMat ** 2
    resultMat = diffMat.sum(axis=1)
    resultMat = resultMat ** 0.5
    index = resultMat.argsort()
    classCount = {}
    for i in range(k):
        voteLabel = labels[index[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0)+1
    sortedClass = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClass[0][0]


def img2Vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2Vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2Vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print "the classifier came back with %d, the real answer is %d" % (classifierResult,classNumStr)
        if classifierResult != classNumStr:
            errorCount += 1
    print 'the total error rate is %f' % (float(errorCount)/float(mTest))

if __name__ == "__main__": 
    handwritingClassTest()