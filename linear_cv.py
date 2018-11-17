from __future__ import print_function
import tensorflow as tf
import numpy as np
import pandas as pd
import Data_util
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

LoadModel = False

experiment_name = 'salary'

data = Data_util.read_data("data/adult.data")
training_data, training_labels = Data_util.class2vect(data)

split_ratio = 0.8
X_train, X_test = training_data[:int(len(training_data)*split_ratio)], training_data[int(len(training_data)*split_ratio):]
y_train, y_test = training_labels[:int(len(training_labels)*split_ratio)], training_labels[int(len(training_labels)*split_ratio):]


reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print('Coefficients: \n', reg.coef_)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print('Variance score: %.2f' % r2_score(y_test, y_pred))
