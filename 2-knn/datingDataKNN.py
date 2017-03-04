from numpy import *
import operator

def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis=1)
    distances = sqDistance ** 0.5
    sortedDistanceIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistanceIndicies[i]]
        classCount[voteLabel] = classCount.get(voteLabel,0)+1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    arrayLines = fr.readlines()
    numberOfLines = len(arrayLines)
    returnMat = zeros((numberOfLines, 3))
    classLabel = []
    index = 0
    for line in arrayLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabel.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabel

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    range = maxVals - minVals
    m = dataSet.shape[0]
    normDataSet = zeros(dataSet.shape)
    normDataSet = (dataSet - tile(minVals, (m, 1)))/tile(range, (m, 1))
    return normDataSet, range, minVals

def datingClassTest():
    hoRatio = 0.10
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],
                                     normMat[numTestVecs:m, :],
                                     datingLabels[numTestVecs:m], 3)
        print "the classifier came back with %d, the real answer is %d" % (classifierResult, datingLabels[i])
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print "the total error rate is: %f" % (errorCount/float(numTestVecs))

def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    ffMiles = float(raw_input('frequent fliter miles earned per year?'))
    percentTats = float(raw_input("percentage of time spent palying video games?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,range,minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats,iceCream])
    classifierResult = classify0((inArr-minVals)/range,normMat,datingLabels,3)
    print "You will probably like this person: ", resultList[classifierResult-1]

if __name__=="__main__":
    classifyPerson()


