"""
Test a learner.  (c) 2015 Tucker Balch
"""

import numpy as np
import math
import LinRegLearner as lrl
from DTLearner import DTLearner
import sys
from RTLearner import RTLearner
from BagLearner import BagLearner
import random
from InsaneLearner import InsaneLearner

def fake_seed(*args):
    pass
def fake_rseed(*args):
    pass

if __name__=="__main__":
    if len(sys.argv) != 2:
        print "Usage: python testlearner.py <filename>"
        sys.exit(1)
    inf = open(sys.argv[1])
    data = np.genfromtxt(inf,delimiter=',')
    if 'Istanbul.csv' in sys.argv[1]:
        alldata = data[1:,1:]
    #data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()])
    # compute how much of the data is training and testing
    # train_rows = int(0.6* data.shape[0])
    # test_rows = data.shape[0] - train_rows

    # # separate out training and testing data
    # trainX = data[:train_rows,0:-1]
    # trainY = data[:train_rows,-1]
    # testX = data[train_rows:,0:-1]
    # testY = data[train_rows:,-1]
    np.random.seed = fake_seed
    random.seed = fake_rseed
    np.random.seed(1481090001)
    random.seed(1481090001)


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
    print testX.shape
    print testY.shape

    testdata= np.array([
    [0.61, 0.63, 8.4, 3],
    [0.885, 0.33, 9.1, 4],
    [0.56, 0.5, 9.4, 6],
    [0.735, 0.57, 9.8, 5],
    [0.32, 0.78, 10, 6],
    [0.26, 0.63, 11.8, 8],
    [0.5, 0.68, 10.5, 7],
    [0.725, 0.39, 10.9, 5],
    ])

        # create a learner and train it
    learner = InsaneLearner(verbose = True) # create a LinRegLearner
    learner.addEvidence(trainX, trainY) # train it
    print learner.author()

    # evaluate in sample
    predY = learner.query(trainX) # get the predictions
    rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
    print
    print "In sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=trainY)
    print "corr: ", c[0,1]

    # evaluate out of sample
    predY = learner.query(testX) # get the predictions
    rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
    print
    print "Out of sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=testY)
    print "corr: ", c[0,1]

    # learner = DTLearner(leaf_size=1, verbose = True) # create a LinRegLearner
    # learner.addEvidence(trainX, trainY) # train it
    # print learner.author()

    # # evaluate in sample
    # predY = learner.query(trainX) # get the predictions
    # rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
    # print
    # print "In sample results"
    # print "RMSE: ", rmse
    # c = np.corrcoef(predY, y=trainY)
    # print "corr: ", c[0,1]

    # # evaluate out of sample
    # predY = learner.query(testX) # get the predictions
    # rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
    # print
    # print "Out of sample results"
    # print "RMSE: ", rmse
    # c = np.corrcoef(predY, y=testY)
    # print "corr: ", c[0,1]

    # learner = BagLearner(learner=RTLearner,kwargs={"leaf_size":1},bags=2,boost=False,verbose=False)
    # learner.addEvidence(trainX,trainY)

    # # evaluate in sample
    # predY = learner.query(trainX) # get the predictions
    # rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
    # print
    # print "In sample results"
    # print "RMSE: ", rmse
    # c = np.corrcoef(predY, y=trainY)
    # print "corr: ", c[0,1]

    # # evaluate out of sample
    # predY = learner.query(testX) # get the predictions
    # rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
    # print
    # print "Out of sample results"
    # print "RMSE: ", rmse
    # c = np.corrcoef(predY, y=testY)
    # print "corr: ", c[0,1]

    # # create a learner and train it
    # learner = lrl.LinRegLearner(verbose = True) # create a LinRegLearner
    # learner.addEvidence(trainX, trainY) # train it
    # print learner.author()

    # # evaluate in sample
    # predY = learner.query(trainX) # get the predictions
    # rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
    # print
    # print "In sample results"
    # print "RMSE: ", rmse
    # c = np.corrcoef(predY, y=trainY)
    # print "corr: ", c[0,1]

    # # evaluate out of sample
    # predY = learner.query(testX) # get the predictions
    # rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
    # print
    # print "Out of sample results"
    # print "RMSE: ", rmse
    # c = np.corrcoef(predY, y=testY)
    # print "corr: ", c[0,1]
