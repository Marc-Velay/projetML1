import numpy as np
import Data_util
import prince
from sklearn import model_selection
from sklearn.cross_decomposition import CCA


data = Data_util.read_data("data/adult.data")
training_data, training_labels = Data_util.class2vect(data)
X_train, X_test, y_train, y_test = model_selection.train_test_split(training_data, training_labels, train_size=0.7, test_size=0.3)


pca = prince.PCA(  n_components=70,
                    n_iter=3,
                    copy=True,
                    rescale_with_mean=True,
                    rescale_with_std=True,
                    engine='auto',
                    random_state=42)

pca = pca.fit(X_train)

print([100*ei for ei in pca.explained_inertia_])
print(sum(pca.explained_inertia_))
print(pca.row_coordinates(X_test[:5]))
print(pca)
