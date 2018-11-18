import tensorflow as tf
import numpy as np
import Data_util
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
import time
import prince

import os
import pickle


LoadData = True
DataFile = 'data/saved_data.pkl'
LabelFile = 'data/saved_labels.pkl'
DataFile_test = 'data/saved_test_data.pkl'
LabelFile_test = 'data/saved_test_labels.pkl'

if LoadData is False or not os.path.isfile(DataFile):
    data = Data_util.read_data("data/adult.data")
    data = Data_util.normalise_data_pandas(data, ["Age", "fnlwgt", "Education-Num", "Capital Gain", "Capital Loss", "Hours per week"])
    training_data, training_labels = Data_util.class2vect(data, one_hot=True)

    data_test = Data_util.read_data("data/adult.test")
    data_test = Data_util.normalise_data_pandas(data_test, ["Age", "fnlwgt", "Education-Num", "Capital Gain", "Capital Loss", "Hours per week"])
    test_data, test_labels = Data_util.class2vect(data_test, one_hot=True)

    if not os.path.isfile(DataFile):
        with open(DataFile, "wb") as f:
            pickle.dump(training_data, f)
    if not os.path.isfile(LabelFile):
        with open(LabelFile, "wb") as f:
            pickle.dump(training_labels, f)

    if not os.path.isfile(DataFile_test):
        with open(DataFile_test, "wb") as f:
            pickle.dump(test_data, f)
    if not os.path.isfile(LabelFile_test):
        with open(LabelFile_test, "wb") as f:
            pickle.dump(test_labels, f)
else:
    if os.path.isfile(DataFile) and os.path.isfile(LabelFile):
        with open(DataFile, "rb") as f:
            training_data = pickle.load(f)
        with open(LabelFile, "rb") as f:
            training_labels = pickle.load(f)
        with open(DataFile_test, "rb") as f:
            test_data = pickle.load(f)
        with open(LabelFile_test, "rb") as f:
            test_labels = pickle.load(f)

X_train, X_val, y_train, y_val = model_selection.train_test_split(training_data, training_labels, train_size=.99, test_size=.01)


print("training!")
n_est_range = range(100, 1100, 500)
max_depth_range = range(5, 25, 10)
param_grid = dict(n_estimators=n_est_range, max_depth=max_depth_range)
cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
grid = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=cv, n_jobs=7, verbose=1)
grid.fit(X_train, y_train)


print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))
