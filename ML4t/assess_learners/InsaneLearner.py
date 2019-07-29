import numpy as np
from BagLearner import BagLearner
from LinRegLearner import LinRegLearner

class InsaneLearner(object):

    def __init__(self, verbose = False):
        self.verbose = verbose

    def author(self):
        return 'xhuang343' # replace tb34 with your Georgia Tech username

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        self.learner = BagLearner(learner=BagLearner, kwargs={"learner":LinRegLearner,"kwargs":{"verbose":False},"bags":20,"boost":False,"verbose":False}, bags=20, boost=False, verbose=False)
        self.learner.addEvidence(dataX,dataY)
        
    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        return self.learner.query(points)


if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
