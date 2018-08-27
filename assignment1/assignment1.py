import loadData
import pandas
import numpy
from sklearn import tree
import graphviz

adult_training_set, adult_testing_set = loadData.loadDataWithTestSet('adult.data', 'adult.test')
hd_training_set, hd_testing_set = loadData.loadDataWithoutTestSet('processed.cleveland.data', True)
print('Adult datasets:')
print(adult_training_set.head())
print(adult_testing_set.head())
print('Heart disease dataset:')
print(hd_training_set.head())
print(hd_testing_set.head())

adult_training_set_labels = adult_training_set.iloc[:,-1]
adult_training_set = adult_training_set.iloc[:,0:-1]
adult_training_set_labels[adult_training_set_labels.str.contains('>50K',na=False)] = 1
adult_training_set_labels[adult_training_set_labels.str.contains('<=50K',na=False)] = 0

adult_testing_set_labels = adult_testing_set.iloc[:,-1]
hd_training_set_labels = hd_training_set.iloc[:,-1]
hd_testing_set_labels = hd_testing_set.iloc[:,-1]
print('Adult datasets labels:')
print(adult_training_set.head())
print(adult_training_set_labels.head())
print(adult_testing_set_labels.head())
print('Heart disease datasets labels:')
print(hd_training_set_labels.head())
print(hd_testing_set_labels.head())

print(adult_training_set_labels[(adult_training_set_labels!=0) & (adult_training_set_labels!=1)])

# decision tree
dtc = tree.DecisionTreeClassifier()
dtc = dtc.fit(adult_training_set, adult_training_set_labels)
dot_data = tree.export_graphviz(dtc, out_file=None)
graph = graphviz.Source(dot_data)
graph.render('adult')