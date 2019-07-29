import numpy as np
import pandas as pd
import math
import LinRegLearner as lrl
from DTLearner import DTLearner
import sys
from RTLearner import RTLearner
from BagLearner import BagLearner
import random
from InsaneLearner import InsaneLearner
import time
import matplotlib.pyplot as plt

def fake_seed(*args):
    pass
def fake_rseed(*args):
    pass

if __name__=="__main__":
    if len(sys.argv) != 2:
        print "Usage: python GraphLearner.py <filename>"
        sys.exit(1)
    inf = open(sys.argv[1])
    data = np.genfromtxt(inf,delimiter=',')
    if 'Istanbul.csv' in sys.argv[1]:
        alldata = data[1:,1:]

    datasize = alldata.shape[0]
    cutoff = int(datasize*0.6)
    permutation = np.random.permutation(alldata.shape[0])
    col_permutation = np.random.permutation(alldata.shape[1]-1)
    train_data = alldata[permutation[:cutoff],:]
    # trainX = train_data[:,:-1]
    trainX = train_data[:,col_permutation]
    trainY = train_data[:,-1]
    test_data = alldata[permutation[cutoff:],:]
    # testX = test_data[:,:-1]
    testX = test_data[:,col_permutation]
    testY = test_data[:,-1]

    create a learner and train it
    result=[]
    leafSizes=[]
    for i in range(50):
        learner = DTLearner(leaf_size=i+1, verbose = True) # create a LinRegLearner
        learner.addEvidence(trainX, trainY) # train it
        predY = learner.query(trainX) # get the predictions
        rmseTrain = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
        cTrain = np.corrcoef(predY, y=trainY)
        predY = learner.query(testX) # get the predictions
        rmseTest = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
        cTest = np.corrcoef(predY, y=testY)
        result.append([rmseTrain, cTrain[0,1], rmseTest, cTest[0,1]])
        leafSizes.append(i+1)
    result = np.array(result)
    leafSizes = np.array(leafSizes)
    print result
    print leafSizes
    plt.plot(leafSizes, result[:,0])
    plt.plot(leafSizes, result[:,2])
    plt.legend(['Training Set', 'Testing Set'])
    plt.title('RMSE Comparison between Training and Testing Datasets')
    plt.xlabel('Leaf Size')
    plt.grid(True)
    plt.ylabel('RMSE')
    plt.savefig('plot1.png')
    plt.clf()
    plt.plot(leafSizes, result[:,1])
    plt.plot(leafSizes, result[:,3])
    plt.legend(['Training Set', 'Testing Set'])
    plt.grid(True)
    plt.title('Correlation Comparison between Training and Testing Datasets')
    plt.xlabel('Leaf Size')
    plt.ylabel('Correlation')
    plt.savefig('plot2.png')

    result=[]
    leafSizes=[]
    for i in range(50):
        learner = BagLearner(learner=DTLearner,kwargs={"leaf_size":i+1},bags=20,boost=False,verbose=False)
        learner.addEvidence(trainX,trainY)
        predY = learner.query(trainX) # get the predictions
        rmseTrain = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
        cTrain = np.corrcoef(predY, y=trainY)
        predY = learner.query(testX) # get the predictions
        rmseTest = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
        cTest = np.corrcoef(predY, y=testY)
        result.append([rmseTrain, cTrain[0,1], rmseTest, cTest[0,1]])
        leafSizes.append(i+1)
    result = np.array(result)
    leafSizes = np.array(leafSizes)
    plt.plot(leafSizes, result[:,0])
    plt.plot(leafSizes, result[:,2])
    plt.legend(['Training Set', 'Testing Set'])
    plt.title('RMSE Comparison between Training and Testing Datasets')
    plt.xlabel('Leaf Size')
    plt.grid(True)
    plt.ylabel('RMSE')
    plt.savefig('plot3.png')
    plt.clf()
    plt.plot(leafSizes, result[:,1])
    plt.plot(leafSizes, result[:,3])
    plt.legend(['Training Set', 'Testing Set'])
    plt.grid(True)
    plt.title('Correlation Comparison between Training and Testing Datasets')
    plt.xlabel('Leaf Size')
    plt.ylabel('Correlation')
    plt.savefig('plot4.png')

    resultDTL=[]
    resultRTL=[]
    leafSizes=[]
    for i in range(50):
        learner = DTLearner(leaf_size=i+1, verbose = True)
        start = time.time()
        learner.addEvidence(trainX,trainY)
        end = time.time()
        buildTreeTime = end-start
        start = time.time()
        predY = learner.query(testX) # get the predictions
        end = time.time()
        queryTime = end-start
        rmseTest = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
        cTest = np.corrcoef(predY, y=testY)
        resultDTL.append([rmseTest, cTest[0,1], buildTreeTime, queryTime])

        learner = RTLearner(leaf_size=i+1, verbose = True)
        start = time.time()
        learner.addEvidence(trainX,trainY)
        end = time.time()
        buildTreeTime = end-start
        start = time.time()
        predY = learner.query(testX) # get the predictions
        end = time.time()
        queryTime = end-start
        rmseTest = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
        cTest = np.corrcoef(predY, y=testY)
        resultRTL.append([rmseTest, cTest[0,1], buildTreeTime, queryTime])
        leafSizes.append(i+1)
    resultDTL = np.array(resultDTL)
    resultRTL = np.array(resultRTL)
    leafSizes = np.array(leafSizes)
    plt.plot(leafSizes, resultDTL[:,0])
    plt.plot(leafSizes, resultRTL[:,0])
    plt.legend(['DTLearner', 'RTLearner'])
    plt.title('RMSE Comparison between DTLearner and RTLearner')
    plt.xlabel('Leaf Size')
    plt.grid(True)
    plt.ylabel('RMSE')
    plt.savefig('plot5.png')
    plt.clf()
    plt.plot(leafSizes, resultDTL[:,1])
    plt.plot(leafSizes, resultRTL[:,1])
    plt.legend(['DTLearner', 'RTLearner'])
    plt.grid(True)
    plt.title('Correlation Comparison between DTLearner and RTLearner')
    plt.xlabel('Leaf Size')
    plt.ylabel('Correlation')
    plt.savefig('plot6.png')
    plt.clf()
    plt.plot(leafSizes, resultDTL[:,2])
    plt.plot(leafSizes, resultRTL[:,2])
    plt.legend(['DTLearner', 'RTLearner'])
    plt.title('Model Building Time between DTLearner and RTLearner')
    plt.xlabel('Leaf Size')
    plt.grid(True)
    plt.ylabel('Time in Seconds')
    plt.savefig('plot7.png')
    plt.clf()
    plt.plot(leafSizes, resultDTL[:,3])
    plt.plot(leafSizes, resultRTL[:,3])
    plt.legend(['DTLearner', 'RTLearner'])
    plt.grid(True)
    plt.title('Query Time between DTLearner and RTLearner')
    plt.xlabel('Leaf Size')
    plt.ylabel('Time in Seconds')
    plt.savefig('plot8.png')
    print resultDTL[:,2]/resultRTL[:,2]
    plt.clf()
    plt.plot(leafSizes, resultDTL[:,2]/resultRTL[:,2])
    plt.grid(True)
    plt.xlabel('Leaf Size')
    plt.ylabel('Multiples of Time')
    plt.savefig('plot9.png')
