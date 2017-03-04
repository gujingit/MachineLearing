#coding=utf-8
from numpy import *


def loadDataSet(fileName):
    dataMat =[]
    fr = open(fileName)
    for line in fr.readlines():
        currLine = line.strip().split('\t')
        fltLine = map(float,currLine)
        dataMat.append(fltLine)
    return dataMat

def distEclud(vecA,vecB):
    return sqrt(sum(power(vecA-vecB,2)))


def randCent(dataSet,k):
    #n = len(dataSet) ##??
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))
    for j in xrange(n):
        minJ = min(dataSet[:,j])
        maxJ = max(dataSet[:,j])
        range = float(maxJ-minJ)
        centroids[:,j] = minJ + range*random.rand(k,1)
    return centroids

def kMeans(dataSet,k,distMeas=distEclud,createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2))) #clusterAssment[0]表示所属类别k，[1]表示到聚簇中心距离
    centroids = createCent(dataSet,k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged=False
        for i in range(m):
            minDist = inf; minIndex =-1
            for j in range(k):
                distJI = distMeas(dataSet[i],centroids[j])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
                clusterAssment[i,0] = minIndex
                clusterAssment[i,1] = minDist**2
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
            centroids[cent,:] = mean(ptsInClust,0)
        return centroids,clusterAssment


def biKmeans(dataSet,k,disMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataSet,0).tolist()[0]
    centList = [centroid0]
    for j in range(m):
        clusterAssment[j,1] = disMeas(mat(centroid0),dataSet[j,:])**2
    while (len(centList)<k):
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]
            centroidMat,splitClustAss = kMeans(ptsInCurrCluster,2,disMeas)
            sseSplit = sum(splitClustAss[:,1])
            sseNoSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            print "sseSplit and not Split",sseSplit,sseNoSplit
            if(sseSplit + sseNoSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClusterAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNoSplit
        bestClusterAss[nonzero(bestClusterAss[:,0].A==1)[0],0] = len(centList) #新增的簇类
        bestClusterAss[nonzero(bestClusterAss[:,0].A==0)[0],0] = bestCentToSplit #本来的簇类
        #更新聚类中心的值
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]
        centList.append(bestNewCents[1,:].tolist()[0])
        #print 'best',bestClusterAss
        clusterAssment[nonzero(clusterAssment[:,0].A==bestCentToSplit)[0],:]=bestClusterAss
        #print clusterAssment
        print len(clusterAssment)
    return mat(centList),clusterAssment

if __name__=="__main__":
    datMat3= mat(loadDataSet('testSet2.txt'))
    centList,myNewAssments=biKmeans(datMat3,3)







