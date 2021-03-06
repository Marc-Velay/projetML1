from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC

import tensorflow as tf
import numpy as np
import Data_util
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
import time
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
#boosting = GradientBoostingClassifier(n_estimators=250, loss='exponential', learning_rate=0.2)
#boosting = AdaBoostClassifier(RandomForestClassifier(n_estimators=600, max_depth=15, n_jobs=7), n_estimators=25, learning_rate=.1)
boosting = AdaBoostClassifier(SVC(C=50, gamma=2), n_estimators=25, learning_rate=.1)
boosting.fit(X_train, y_train[:,1])

print("Classifier has a score of %0.4f"
      % (boosting.score(test_data, test_labels[:,1])))
