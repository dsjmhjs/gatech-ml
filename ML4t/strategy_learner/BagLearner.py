"""
A simple wrapper for linear regression.  (c) 2015 Tucker Balch
"""
# Xicheng Huang xhuang343

import numpy as np
import pandas as pd
import scipy.stats as sc
import RTLearner as rtl
# import DTLearner as dtl

class BagLearner(object):

    def __init__(self, learner, bags, boost = False, verbose = False, kwargs={}):
        self.learner = learner
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        self.kwargs = kwargs
        

    def author(self):
        return 'xhuang343' # replace tb34 with your Georgia Tech username

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        num=int(dataX.shape[0]*0.6)
        self.learners = {}
        for i in range(self.bags):
            self.learners['lr{}'.format(i)] = self.learner(**self.kwargs)
            rand=np.random.randint(dataX.shape[0], size=num)
            self.learners['lr{}'.format(i)].addEvidence(dataX[rand,:], dataY[rand])
        
    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        sum=[]
        for i in range(self.bags):
            sum.append(self.learners['lr{}'.format(i)].query(points))
        result=np.nanmean(sum, axis=0)
        result[result >= 0.3] = 1
        result[result <= -0.3] = -1
        result[((result > - 0.3) & (result < 0.3))] = 0
        return result


if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
