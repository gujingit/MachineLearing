#coding=utf-8
from numpy import *


def loadDataSet():
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]


def createC1(dataSet):
    C1=[]
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return map(frozenset,C1) #frozenset不可变


def scanD(D,Ck,minSupport): #D 数据集 Ck频繁项集 minSupport最小支持度
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not ssCnt.has_key(can):
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))
    retList =  []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.insert(0,key)
            supportData[key] = support
    return retList,supportData


def aprioriGen(Lk,k):
    retList = [] # 返回频繁项集
    lenLk = len(Lk)
    for i in range(lenLk): #  根据上一层的频繁项集生成下一层的
        for j in range(i+1,lenLk):
            L1 = list(Lk[i])[:k-2] #列表从0到(k-2)-1 当k=2时,L1为[]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1==L2:
                retList.append(Lk[i]|Lk[j]) # | 集合的并集
    return retList


def apriori(dataSet,minSupport=0.5): #挖掘频繁项集
    C1 = createC1(dataSet)
    D = map(set,dataSet)
    L1,supportData = scanD(D,C1,minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2])>0):
        Ck = aprioriGen(L[k-2],k)
        Lk,supK = scanD(D,Ck,minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData


def generateRules(L,supportData,minConf = 0.7): #挖掘关联规则
    bigRuleList=[]
    for i in range(1,len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i >1):
                rulesFromConseq(freqSet,H1,supportData,bigRuleList,minConf)
            else:
                calcConf(freqSet,H1,supportData,bigRuleList,minConf)
    return bigRuleList


def calcConf(freqSet,H,supportData,brl,minConf =0.7):
    prunedH =[]
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq] # p->H,conf = suppport(p|H)/support(p) '|' set or
        if conf>=minConf:
            print freqSet-conseq,'-->',conseq,'conf',conf
            brl.append((freqSet-conseq,conseq,conf))
            prunedH.append(conseq)
    return prunedH


def rulesFromConseq(freqSet,H,supportData,brl,minConf = 0.7): #规则合并[2,3,5->1] 计算[2,3->5,1]
    m = len(H[0])
    if (len(freqSet) > (m+1)):
        Hmp1 = aprioriGen(H,m+1)
        Hmp1 = calcConf(freqSet,Hmp1,supportData,brl,minConf)
        print Hmp1
        if (len(Hmp1)>1): #Hmp1  不为空
            rulesFromConseq(freqSet,Hmp1,supportData,brl,minConf)




def mushroom():
    mushDatSet = [line.split() for line in open('mushroom.dat').readlines()]
    L,supportData = apriori(mushDatSet,minSupport=0.3)
    print len(L[1])
    #rules = generateRules(L,supportData,minConf=0.7)




if __name__=='__main__':
    dataSet = loadDataSet()
    L,supportData = apriori(dataSet)
    rules = generateRules(L,supportData,minConf=0.5)
    print rules

    #mushroom()


