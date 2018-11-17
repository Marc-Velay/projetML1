import numpy as np
import Data_util
from sklearn import model_selection
from sklearn import metrics
from sklearn.utils import class_weight
import os
import pickle
import keras_callbacks

from matplotlib import pyplot as plt

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam


experiment_name = 'salary'
TRAIN = True
model_filename = 'save/MLP2.h5'

def build_model(input_dim, nb_targets):
    model = Sequential()
    model.add(Dense(2048, input_shape=(input_dim)))
    model.add(LeakyReLU(alpha=.001))
    model.add(Dropout(0.3))
    model.add(Dense(2048))
    model.add(LeakyReLU(alpha=.001))
    model.add(Dropout(0.3))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=.001))
    model.add(Dropout(0.3))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=.001))
    model.add(Dense(64))
    model.add(LeakyReLU(alpha=.01))
    model.add(Dense(nb_targets))
    return model
'''def build_model(input_dim, nb_targets):
    model = Sequential()
    model.add(Dense(2048, input_shape=(input_dim)))
    model.add(LeakyReLU(alpha=.01))
    model.add(Dropout(0.3))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=.01))
    model.add(Dropout(0.3))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=.01))
    model.add(Dropout(0.3))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=.01))
    model.add(Dense(64))
    model.add(LeakyReLU(alpha=.01))
    model.add(Dense(nb_targets))
    return model'''

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


X_train, X_val, y_train, y_val = model_selection.train_test_split(training_data, training_labels, train_size=0.7, test_size=0.3)

class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train[:,1]),
                                                 y_train[:,1])

optimizer = Adam()
MLP = build_model(X_train[0].shape, 2)
MLP.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


if TRAIN:
    histories = keras_callbacks.Histories()
    checkpoint = ModelCheckpoint(model_filename, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint, histories]

    history = MLP.fit(X_train, y_train, validation_data=(X_val, y_val),
    	epochs=25, class_weight=class_weights, callbacks=callbacks_list, batch_size=16)

    plt.figure(1)
    plt.plot(history.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    plt.show()


MLP.load_weights(model_filename)

predictions = MLP.predict(test_data)
matrix = metrics.confusion_matrix(test_labels.argmax(axis=1), predictions.argmax(axis=1))
print(matrix)
