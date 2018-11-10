import tensorflow as tf
import numpy as np
import Layers
import Data_util
import pylab as pl
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from numpy import array
import pandas as pd
import operator

LoadModel = False

experiment_name = 'salary'

data = Data_util.read_data("data/adult.data")
training_data, training_labels = Data_util.class2vect(data)

split_ratio = 0.8
X_train, X_test = training_data[:int(len(training_data)*split_ratio)], training_data[int(len(training_data)*split_ratio):]
y_train, y_test = training_labels[:int(len(training_labels)*split_ratio)], training_labels[int(len(training_labels)*split_ratio):]

k_List = list(range(1,50))
neighbors = filter(lambda x: x % 2 != 0, k_List)
cv_scores = []

for k in neighbors:
	knn = KNeighborsClassifier(n_neighbors = k)
	scores = cross_val_score(knn, X_train, np.ravel(y_train), cv = 10, scoring = 'accuracy')
	cv_scores.append(scores.mean())

MSE = [1 - x for x in cv_scores]
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)
print("With a mean square error of : ", min(MSE))
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()

exit(0)
