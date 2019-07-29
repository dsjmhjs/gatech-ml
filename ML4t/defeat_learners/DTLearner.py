"""
A simple wrapper for linear regression.  (c) 2015 Tucker Balch
"""

import numpy as np
import pandas as pd
import scipy.stats as sc

class DTLearner(object):

    def __init__(self, leaf_size=1, verbose = False):
        self.leaf_size = leaf_size # move along, these aren't the drones you're looking for

    def author(self):
        return 'xhuang343' # replace tb34 with your Georgia Tech username

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        self.tree = self.build_tree(dataX, dataY)
        
        
    def build_tree(self, dataX, dataY):
        #print dataX.shape
        if dataX.shape[0]<=self.leaf_size:
            value = np.mean(dataY)
            return np.array([-1, value, None, None])
        elif np.unique(dataY).size == 1:
            return np.array([-1, dataY[0], None, None])
        else:
            maxCorr = 0
            corr = 0
            bestFeature = 0
            if dataX.shape[0] > 2:
                for i in range(dataX.shape[1]):
                    if np.unique(dataX[:,i]).size > 1:
                        corr = abs(sc.pearsonr(dataX[:,i], dataY)[0])
                    if corr > maxCorr:
                        maxCorr = corr
                        bestFeature = i
            else:
                bestFeature = 0
            SplitVal = np.median(dataX[:,bestFeature])
            leftTreeCond = dataX[:,bestFeature]<=SplitVal
            rightTreeCond = dataX[:,bestFeature]>SplitVal
            if len(leftTreeCond)>0 and all(leftTreeCond)==True:
                leftTreeCond[0]=False
                rightTreeCond[0]=True
            leftTree = self.build_tree(dataX[leftTreeCond], dataY[leftTreeCond])
            rightTree = self.build_tree(dataX[rightTreeCond], dataY[rightTreeCond])
            if leftTree.ndim == 1:
                leftTreeShape = 1
            else:
                leftTreeShape = leftTree.shape[0]
            root=np.array([bestFeature, SplitVal, 1, leftTreeShape+1])
            return np.vstack((root, leftTree, rightTree))

    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        np.set_printoptions(threshold=np.inf)
        result = []
        #print self.tree
        for i in range(points.shape[0]):
            j=0
            while True:
                if self.tree[j,0] == -1:
                    result.append(self.tree[j,1])
                    break;
                if points[i,int(self.tree[j,0])] <= self.tree[j,1]:
                    j+= int(self.tree[j,2])
                else:
                    j+= int(self.tree[j,3])
        return result

if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
