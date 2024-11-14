# from ucimlrepo import fetch_ucirepo 
# import matplotlib

# import arff, numpy as np

from sklearn.datasets import fetch_openml, load_wine
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------Wine-------------------------

# 1. import data
ds = pd.read_csv('CongressionalVotingID.shuf.lrn.csv')
# ds = fetch_openml(name='mushroom', version=1)
# print(ds)
print(ds.sample(3))
X = ds.drop(['ID', 'class'], axis=1)
y = ds['class']
# print(X)
# print(y)

# 2. data exploration and preprocessing
# print(wine) 
# print(wine.data)
# print(wine.target)
# print(wine.DESCR)
enc = Pipeline(steps=[
    ("encoder", preprocessing.OrdinalEncoder()),
    # ("encoder", preprocessing.OneHotEncoder()),
    ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
    ("scaler", preprocessing.StandardScaler()),
])
X_preprocessed = enc.fit_transform(X)
# scaler = preprocessing.StandardScaler().fit(X)
# X_scaled = scaler.transform(X)
X_scaled = X_preprocessed


# 3. classification
# X = [[0., 0.], [1., 1.]]
# y = [0, 1]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(15, 2), 
                    random_state=1)
# clf.fit(X, y)
clf.fit(X_scaled, y)


# 4. performance evaluation
# accuracy & precision, false positives, false negatives

#holdout
X_train, X_test, y_train, y_test = train_test_split(
    # X, y, test_size=0.4, random_state=0)
    X_scaled, y, test_size=0.4, random_state=0)

# X_train.shape, y_train.shape
# X_test.shape, y_test.shape

# clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    # hidden_layer_sizes=(15, 2), 
                    # random_state=1)
# clf.fit(X, y)
clf.fit(X_scaled, y)

# clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
print(clf.score(X_test, y_test))
print("accurancy from holdout\n")

#crossvalidation
# clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    # hidden_layer_sizes=(15, 2), 
                    # random_state=1)
# scores = cross_val_score(clf, X, y, cv=10)
scores = cross_val_score(clf, X_scaled, y, cv=10)
print(scores)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


#some visulization?

