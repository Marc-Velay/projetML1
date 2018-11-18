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


experiment_name = 'salary'

LoadData = True
DataFile = 'data/saved_data.pkl'
LabelFile = 'data/saved_labels.pkl'

if LoadData is False or not os.path.isfile(DataFile):
    data = Data_util.read_data("data/adult.data")
    data = Data_util.normalise_data_pandas(data, ["Age", "fnlwgt", "Education-Num", "Capital Gain", "Capital Loss", "Hours per week"])
    training_data, training_labels = Data_util.class2vect(data, one_hot=True)
    if not os.path.isfile(DataFile):
        with open(DataFile, "wb") as f:
            pickle.dump(training_data, f)
    if not os.path.isfile(LabelFile):
        with open(LabelFile, "wb") as f:
            pickle.dump(training_labels, f)
else:
    if os.path.isfile(DataFile) and os.path.isfile(LabelFile):
        with open(DataFile, "rb") as f:
            training_data = pickle.load(f)
        with open(LabelFile, "rb") as f:
            training_labels = pickle.load(f)


X_train, X_test, y_train, y_test = model_selection.train_test_split(training_data, training_labels, train_size=0.7, test_size=0.3)


print("training!")
n_est_range = range(100, 1100, 500)
max_depth_range = range(5, 25, 10)
param_grid = dict(n_estimators=n_est_range, max_depth=max_depth_range)
cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
grid = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=cv, n_jobs=7, verbose=1)
grid.fit(X_train, y_train)


print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))
