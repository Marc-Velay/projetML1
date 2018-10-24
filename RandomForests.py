import tensorflow as tf
import numpy as np
import Data_util
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
import time

experiment_name = 'salary'

data = Data_util.read_data("data/adult.data")
data = Data_util.normalise_data_pandas(data, ["Age", "fnlwgt", "Education-Num", "Capital Gain", "Capital Loss", "Hours per week"])
training_data, training_labels = Data_util.class2vect(data)

X_train, X_test, y_train, y_test = model_selection.train_test_split(training_data, training_labels, train_size=0.7, test_size=0.3)

#clf = RandomForestClassifier(n_estimators=700, max_depth=25, random_state=55, n_jobs=6)

t0 = time.time()
print("training!")
n_est_range = range(100, 1100, 250)
max_depth_range = range(5, 25, 5)
param_grid = dict(n_estimators=n_est_range, max_depth=max_depth_range)
cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
grid = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=cv, n_jobs=7, verbose=1)
grid.fit(X_train[:50], y_train[:50,1])

print(time.time()-t0)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

#based on best params from GridSearchCV
clf = RandomForestClassifier(n_estimators=600, max_depth=20, random_state=55, n_jobs=7)
clf.fit(X_train, y_train)

print(clf.feature_importances_)


ypred = clf.predict(X_test)
print(ypred[:, 0].shape)
print(np.array(y_test)[:, 0].shape)
err_test = metrics.accuracy_score (ypred[:, 0], np.array(y_test)[:, 0])
print("test set accuracy : %.3f" % err_test)
print(metrics.confusion_matrix(ypred[:, 0], np.array(y_test)[:, 0]))

ypred = clf.predict(X_train)
err_train = metrics.accuracy_score (ypred[:, 0], np.array(y_train)[:, 0])
print("train set accuracy : %.3f" % err_train)
print(metrics.confusion_matrix(ypred[:, 0], np.array(y_train)[:, 0]))
