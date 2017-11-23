from numpy import *
import operator
import matplotlib.pyplot as plt

def creatDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group,labels

def classify0(intX, dataSet, labels, k): 
    dataSetSize = dataSet.shape[0]
    diffMat = tile(intX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum( axis = 1)
    distances = sqDistances**0.5
    sortedDistDices = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistDices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse = True)
    return sortedClassCount[0][0]
def file2matrix(filename):
    fr = open(filename,'r')
    arraylines = fr.readlines()
    linesOfFile = len(arraylines) 
    returnMat = tile(0.0,(linesOfFile,3))
    classLabelVector = []
    index = 0
    for line in arraylines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int (listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = (dataSet - minVals)/ranges
    return normDataSet,ranges,minVals
def datingClassTest():
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    numTestVecs = 100 
    dims = normMat.shape[0]
    k_count = []
    k = 50 
    error = []
    while(numTestVecs > 20):
        errorCount = 0.0
        for i in range(numTestVecs):
            classResult = classify0(normMat[i,:], normMat[numTestVecs:dims,:], datingLabels[numTestVecs:dims],4 )
            if classResult != datingLabels[i]:
                errorCount += 1
        print("when numTestVecs is : %d . the total error rate is:%f"%(numTestVecs ,errorCount/float(numTestVecs)))
        error.append(errorCount/float(numTestVecs))
        k_count.append(numTestVecs)
        numTestVecs -=1 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(k_count,error,'-')
    plt.show()
    
    

if __name__ == '__main__':
#   group,labels = creatDataSet()
#   datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
#   print(autoNorm(datingDataMat))
    datingClassTest()
