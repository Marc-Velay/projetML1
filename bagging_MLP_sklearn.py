import numpy as np
import pandas as pd
import Data_util
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn import model_selection
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
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

bagging = BaggingClassifier(MLPClassifier(hidden_layer_sizes=(256,256,128,64), max_iter=30, alpha=1e-4,
                    solver='adam', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.01), max_samples=0.5, max_features=0.5, n_jobs=7)

bagging.fit(X_train, y_train[:,1])

print("Classifier has a score of %0.4f"
      % (bagging.score(test_data, test_labels[:,1])))
