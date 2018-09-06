## **Supervised Learning - Report**
#### Xicheng Huang (xhuang343) - Fall 2018

## Table of Content
- [Introduction](#introduction)
- [Dataset One](#dataset-one)
    - [Preprocessing](#preprocessing)
    - [Data Investigation](#data-investigation)
    - [Decision Tree](#decision-tree)
    - [Neural Networks](#neural-networks)
    - [Boosting](#boosting)

## Introduction
For this assignment, I picked two completely different datasets in terms of their characteristics for classification problems so that I can learn applying supervised learning algorithms on a more well-rounded way. The first dataset is somewhat a larger dataset where it consists of more than 48000 records. It is a binary classification problem and the features are categorical. The second dataset I picked is a much smaller dataset with only 304 records, and it is also a multi-class problem. For each dataset, I will explain what I did to preprocess them and then go into each of the supervised learning algorithms I applied and their corresponding performance. 

For this assignment, I primarily used Python's scikit-learn library to perform machine learning tasks and used various libraries, including matplotlib and sklearn-evaluation to graph various performance metrics and results of the applied algorithms. 

## Dataset One
### **Preprocessing**
The first dataset is from UC Irvine Machine Learning Repository, and it is called Adult data set. This is an extraction from the 1994 Census database and prediction task is to determine whether a person makes over 50K a year. Once I loaded the data with headers and the right delimiter, without any preprocessing, the dataset looks like this:

![](1.PNG)

As you can seem the dataset is full of categorical features and the label (incomeCategory) is a string representation as well. Because scikit-learn's classification algorithms will not accept any string representation features, one of the first tasks is to convert them to numerical. I could enumerate the categories for each feature, for example, 0 for "Bachelors" and 1 for "HS-grad" in education, but scikit-learn will treat them as continuous data, as a result, the models could provide wrong "understanding" of these features. So the solution is one-hot-encode all of them. After encoding, it will add extra columns for each category of each of the feature. For example, row 0 will have education_Bachelors as 1, meaning it indeed has a education of Bachelors for the feature, and everything other columns for education is 0. Also, I converted the labels into 0s and 1s where 0 is "<=50K" and 1 is ">50K". Then, there were also a handful missing data, so I just set them as 0. After preprocessing, the dataset looks like this:

![](2.PNG)

### **Data Investigation**
After preprocessing them, I randomly divided the dataset up into thirds, 2/3 will be used for training, and 1/3 will be used to testing. Then, I wanted to check the labels to see if the distribution of classes are even. Here is the result:

![](training_set_label_dist.png) ![](testing_set_label_dist.png)
As you can see, there are far more "<=50K" labels in both the training set and testing set. So to make sure the machine learning models are performant, I should not only look at accuracy score but also f1 scores.

### **Decision Tree**
For decision tree classifier, I used scikit-learn's `DecisionTreeClassifier()` to create the model. Without pruning, the tree has more than 9000 nodes! This is very large and definitely susceptible to overfitting. So I used a custom pruning method to prune the tree after modeling. The custom pruning method takes in a threshold parameter and utilize it to check the value of each node and if the smallest value of the node is below that threshold, it will prune it and its children out. Here is a graph representing the benefit of this pruning process:

![](DT1.png)

As you can see, without any pruning, the model is overfitting there it is at almost 100% accuracy when predicting training set while it is not doing so hot for testing set. As the pruning threshold increases, predictions for training set and testing set are converaging, and overfitting fades out. The final tree looks like this:

<img src="tree1.png" height="600px" />

Using a model with pruning threshold of 350, I created a f1 score report.
![](3.PNG)

It is pretty good for determining "<=50K" class but not so good at ">50K". Let's look at the what the next algorithm can do.

### **Neural Networks**
For this algorithm, I used `MLPClassifier()`, and specifically, I used logistic regression in combination of stochastic gradient descent for activation because according to the lecture, calculus is better :). I also incorporated cross-validation and hyperparameter tuning to try to get the best result from this algorithm. For this, I used `GridSearchCV()` provided by scikit-learn. The result is this:

<img src="NN1.png" width="400px" ><img src="NN2.png" width="400px">

As you can see that the accuracy is around 76.3% and could not get any higher as I tune the parameters. Here is the f1 score result I got from the more accurate model:

![](4.png)

Even though the precision for ">50K" is at 1, the recall is at 0, causing f1 score to be very low. Comparatively, the model I was able to produce from neural networks is not as good as the decision tree model above. 

### **Boosting**