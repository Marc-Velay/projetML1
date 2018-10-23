import tensorflow as tf
import numpy as np
import Data_util
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
import time

experiment_name = 'salary'

data = Data_util.read_data("data/adult.data")
data = Data_util.normalise_data_pandas(data, ["Age", "fnlwgt", "Education-Num", "Capital Gain", "Capital Loss", "Hours per week"])
training_data, training_labels = Data_util.class2vect(data)

X_train, X_test, y_train, y_test = model_selection.train_test_split(training_data, training_labels, train_size=0.7, test_size=0.3)

clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=55, n_jobs=6)
t0 = time.time()
clf.fit(X_train, y_train)
print(time.time()-t0)

print(clf.feature_importances_)


ypred = clf.predict(X_test)
print(ypred)
print(y_test)
err_test = metrics.accuracy_score (ypred, y_test, normalize=False)
print("test set accuracy : %.3f" % err_test)
print(metrics.confusion_matrix(y_test, ypred))

ypred = clf.predict(X_train)
err_train = metrics.accuracy_score (ypred, y_train)
print("train set accuracy : %.3f" % err_train)
print(metrics.confusion_matrix(y_train, ypred))
