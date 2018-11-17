from __future__ import print_function
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import GridSearchCV
import Data_util

LoadModel = False

experiment_name = 'salary'

data = Data_util.read_data("data/adult.data")
training_data, training_labels = Data_util.class2vect(data)

split_ratio = 0.8
X_train, X_test = training_data[:int(len(training_data)*split_ratio)], training_data[int(len(training_data)*split_ratio):]
y_train, y_test = training_labels[:int(len(training_labels)*split_ratio)], training_labels[int(len(training_labels)*split_ratio):]


parameters = {'max_depth':range(3,20)}
clf = GridSearchCV(tree.DecisionTreeClassifier(), parameters, n_jobs=7, cv=5)
clf.fit(X=X_train, y=y_train)
tree_model = clf.best_estimator_
print (clf.best_score_, clf.best_params_) 
