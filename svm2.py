from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, confusion_matrix
import Data_util
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit

LoadModel = False

experiment_name = 'salary'

data = Data_util.read_data("data/adult.data")
training_data, training_labels = Data_util.class2vect(data)

split_ratio = 0.8
X_train, X_test = training_data[:int(len(training_data)*split_ratio)], training_data[int(len(training_data)*split_ratio):]
y_train, y_test = training_labels[:int(len(training_labels)*split_ratio)], training_labels[int(len(training_labels)*split_ratio):]

print("Starting cross validation")

C_range = [1]#[1e-2, 1, 1e1]
gamma_range = [1]#[1e-1, 1, 1e1]
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv, verbose=1, n_jobs=6)
grid.fit(X_train, y_train)

print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))
