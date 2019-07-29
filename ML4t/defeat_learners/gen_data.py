"""
template for generating data to fool learners (c) 2016 Tucker Balch
"""

import numpy as np
import math

# this function should return a dataset (X and Y) that will work
# better for linear regression than decision trees
def best4LinReg(seed=1489683273):
    np.random.seed(seed)
    numCol = np.random.randint(low=2,high=1000)
    numRow = np.random.randint(low=10,high=1000)
    X = np.random.rand(numRow, numCol)
    xMultiplier = np.random.randint(10, size=numCol)
    Xprocessed = X * xMultiplier
    Y = np.sum(Xprocessed, axis=1)
    return X, Y

def best4DT(seed=1489683273):
    np.random.seed(seed)
    numCol = np.random.randint(low=2,high=1000)
    numRow = np.random.randint(low=10,high=1000)
    X = np.random.rand(numRow, numCol)
    Y = np.sin(X)[:,0]
    return X, Y

def author():
    return 'xhuang343' #Change this to your user ID

if __name__=="__main__":
    print "they call me Tim."
