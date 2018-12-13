import sklearn


import numpy as np
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score


data_path = './data/pretrained_sentence_emb.pkl'
file = open(data_path, 'rb')
data = pickle.load(file)
X_train = np.squeeze(data['X_train'])
Y_train = data['Y_train']
X_val = np.squeeze(data['X_val'])
Y_val = data['Y_val']
X_test = np.squeeze(data['X_test'])
Y_test = data['Y_test']


print('Fitting Model...')

# Naive Bayes
# model = GaussianNB()

# k-nearest neighbour
# model = KNeighborsClassifier(n_neighbors=20)
# Support Vector Machine
model = SVC(kernel='linear')
# model = SVC()
# model = SVC(C=10)
# model = SVC(C=100)

# CART decision tree
# model = DecisionTreeClassifier()
# model = DecisionTreeClassifier(max_depth=10)

# GBDT
# model = GradientBoostingClassifier()

# MLP
# model = MLPClassifier(hidden_layer_sizes=(256, 256, 2))
#model = MLPClassifier(hidden_layer_sizes=(256, 256, 256, 256, 2))

model.fit(X_train, Y_train)
yhat_train = model.predict(X_train)
yhat_val = model.predict(X_val)
yhat_test = model.predict(X_test)

print('Training Accuarcy: {:.3f}.'.format(accuracy_score(Y_train, yhat_train)))
print('Validation Accuarcy: {:.3f}.'.format(accuracy_score(Y_val, yhat_val)))
print('Testing Accuarcy: {:.3f}.'.format(accuracy_score(Y_test, yhat_test)))