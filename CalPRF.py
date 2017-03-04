from numpy import *

def calPRF(result,predict):
    m = len(result)
    acc = sum(result == predict) / float(m)
    R = sum(result & predict) / float(sum(result == 1))
    P = sum(result & predict) / float(sum(predict == 1))
    F = R * P * 2 / float(P + R)
    print 'acc=', acc, 'R=', R, 'P=', P, 'F=', F

if __name__=='__main__':
    result = array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1])
    predict = array([1, 1, 0, 1, 1, 1, 1, 0, 0, 0])
    calPRF(result,predict)
