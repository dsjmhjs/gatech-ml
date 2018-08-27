import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split


def loadDataWithTestSet(training_set_path, testing_set_path):
    training_set = pd.read_csv(training_set_path)
    testing_set = pd.read_csv(testing_set_path)
    return training_set, testing_set

def loadDataWithoutTestSet(training_set_path, split_to_test=False):
    training_set = pd.read_csv(training_set_path)
    if split_to_test:
        training_set, testing_set = train_test_split(training_set, test_size=(2/3))
        return training_set, testing_set
    return training_set