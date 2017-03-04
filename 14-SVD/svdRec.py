#coding=utf-8
from numpy import *
from numpy import linalg as la


def loadExData():
    return [[1,1,1,2,0],
            [2,2,2,0,0],
            [1,1,1,0,0],
            [5,5,5,0,0],
            [4,4,0,2,2],
            [4,0,0,3,3],
            [4,0,0,1,1]]

def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

#欧几里得距离
def eulidSim(inA,inB):
    return 1.0/(1.0+la.norm(inA-inB))

#Pearson correlation
def pearsSim(inA,inB):
    if len(inA)<3:
        return 1.0
    return 0.5+0.5*corrcoef(inA,inB,rowvar=0)[0][1]

#cos
def cosSim(inA,inB):
    num = float(inA.T*inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom)

def standEst(dataMat,user,simMeas,item):
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n):
        userRating =dataMat[user,j]
        if userRating == 0: continue
        overlap = nonzero(logical_and(dataMat[:,item].A>0,dataMat[:,j].A>0))[0]
        if len(overlap) == 0:
            similarity =0
        else:
            similarity = simMeas(dataMat[overlap,item],dataMat[overlap,j])
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal


def svdEst(dataMat,user,simMeas,item):
    n = shape(dataMat)[1]
    print shape(dataMat)
    simTotal = 0
    ratSimTotal = 0
    u,sigma,vt = la.svd(dataMat)
    Sig4 = mat(eye(4)*sigma[:4])
    xformedItems = dataMat.T * u[:,:4] * Sig4.I # 为什么要用逆? SVD降维就是通过dataMat.T*U*sigma
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0 or item == j: continue
        similarity = simMeas(xformedItems[item,:].T,xformedItems[j,:].T)
        simTotal += similarity
        ratSimTotal += simTotal*userRating
    if simTotal == 0:return 0
    else:
        return ratSimTotal/simTotal

# 图片压缩 从32*32 压缩到32*2+2+32*2  #
def printMat(inMat,thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i,k]) > thresh:
                print 1,
            else:
                print 0,
        print ' '

def imgCompress(numSV=3,thresh=0.8):
    myl = []
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = mat(myl)
    print "***original matrix***"
    printMat(myMat,thresh)

    u,sigma,vt = la.svd(myMat)
    sigRecon = mat(zeros((numSV,numSV)))
    for k in range(numSV):
        sigRecon[k,k]=sigma[k]
    # 图像还原 存储时只需要存储u,sigma,vt矩阵即可
    reconMat = u[:,:numSV]*sigRecon*vt[:numSV,:]
    print "****reconstructed matrix using %d singular values *****" % numSV
    printMat(reconMat,thresh)
# 图片压缩 end #


def recommend(dataMat,user,N=3,simMeas=cosSim,estMethod=standEst):
    unratedItems = nonzero(dataMat[user,:].A==0)[1]
    if len(unratedItems) == 0:
        return 'You have rated everything!'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat,user,simMeas,item)
        itemScores.append((item,estimatedScore))
    return sorted(itemScores,key = lambda jj:jj[1],reverse = True)[:N]

if __name__=="__main__":
    # dataSet = loadExData()
    # u,sigma,vt = la.svd(dataSet)
    # print 'u',u
    # print 'sigma',sigma
    # print 'vt',vt
    # sigma = mat([[sigma[0],0,0],[0,sigma[1],0],[0,0,sigma[2]]])
    # print u[:,:3]*sigma*vt[:3,:]

    dataMat = mat(loadExData())
    result = recommend(dataMat,1,simMeas=cosSim,estMethod=svdEst)
    print result

    #计算90%信息时所需的奇异值数量
    # dataSet = loadExData2()
    # u,sigma,vt = la.svd(dataSet)
    # total = sum(sigma**2)
    # sum = 0
    # index = -1
    # for i in xrange(len(sigma)):
    #     if sum < 0.9*total:
    #         sum += sigma[i]**2
    #     else:
    #         index = i
    #         break
    # print index

    #imgCompress()
