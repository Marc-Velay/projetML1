from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier

from sklearn.neural_network import MLPClassifier
import tensorflow as tf
import numpy as np
import Data_util
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.svm import SVC
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

MLP = BaggingClassifier(MLPClassifier(hidden_layer_sizes=(256,128,128,64), max_iter=30, alpha=1e-4,
                    solver='adam', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.01), max_samples=0.5, max_features=0.8, n_jobs=7)
MLP2 = BaggingClassifier(MLPClassifier(hidden_layer_sizes=(256,256,256,64), max_iter=30, alpha=1e-4,
                    solver='adam', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.01), max_samples=0.5, max_features=0.8, n_jobs=7)

print("training!")
Classifiers = []
Classifiers.append(('RF',RandomForestClassifier(n_estimators=600, max_depth=15)))
Classifiers.append(('MLP',MLP))
Classifiers.append(('MLP2',MLP2))
Classifiers.append(('RF1',AdaBoostClassifier(RandomForestClassifier(n_estimators=400, max_depth=10))))
Classifiers.append(('RF2',BaggingClassifier(RandomForestClassifier(n_estimators=600, max_depth=15), max_samples=0.5, max_features=0.5, n_jobs=7)))
Classifiers.append(('RF3',AdaBoostClassifier(RandomForestClassifier(n_estimators=300, max_depth=5))))
Classifiers.append(('GBC',GradientBoostingClassifier(n_estimators=250, loss='exponential', learning_rate=0.2)))
Classifiers.append(('SVC',AdaBoostClassifier(SVC(C=10, gamma=1), algorithm='SAMME')))


comb = VotingClassifier(estimators=Classifiers, n_jobs=7)
comb.fit(X_train, y_train[:,1])
print('predicting')
print("Classifier has a score of %0.4f"
      % (comb.score(test_data, test_labels[:,1])))
