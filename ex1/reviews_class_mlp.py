# from ucimlrepo import fetch_ucirepo 
# import matplotlib

# import arff, numpy as np

from sklearn.datasets import fetch_openml, load_wine
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pprint import pprint
from ds_load_util import load_dataset

# ----------------Wine-------------------------

# 1. import data
X_train, X_test, y_train, y_test  = load_dataset('reviews')

# 2. data exploration and preprocessing
enc = Pipeline(steps=[
    # ("encoder", preprocessing.OrdinalEncoder()),
    # ("encoder", preprocessing.OneHotEncoder()),
    # ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
    # ("scaler", preprocessing.StandardScaler()),
    # ("scaler", preprocessing.RobustScaler()),
    # ("scaler", preprocessing.MaxAbsScaler()),
    # ("scaler", preprocessing.MinMaxScaler()),
    # ("scaler", preprocessing.StandardScaler(with_mean=False)),
    # ("normalizer", preprocessing.Normalizer()),
])
# X_train = enc.fit_transform(X_train)
# X_test = enc.transform(X_test)
# enc = Pipeline(steps=[
#     # ("encoder", preprocessing.OrdinalEncoder()),
#     # ("encoder", preprocessing.OneHotEncoder()),
#     # ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
#     # ("scaler", preprocessing.StandardScaler()),
#     # ("clf", MLPClassifier(random_state=1)),
# ])
# X_preprocessed = enc.fit_transform(X)
# # scaler = preprocessing.StandardScaler().fit(X)
# # X_scaled = scaler.transform(X)
# X_scaled = X_preprocessed



# parameter tuning
pipeline = Pipeline([
    # ("o_encoder", preprocessing.OrdinalEncoder()),
    # # ("oh_encoder", preprocessing.OneHotEncoder()),
    # # ("imputer", SimpleImputer()),
    # ("imputer", SimpleImputer(strategy='constant',fill_value=-1)),
    # ("scaler", preprocessing.StandardScaler()),
    # # ("scaler", preprocessing.RobustScaler()),
    # # ("scaler", preprocessing.MaxAbsScaler()),
    # # ("scaler", preprocessing.MinMaxScaler()),
    # # ("scaler", preprocessing.StandardScaler(with_mean=False)),
    ("clf", MLPClassifier(random_state=1)),
])

parameter_grid = {
    # "oh_encoder__drop": (None, "if_binary"),
    # "oh_encoder__handle_unknown": ("infrequent_if_exist"),
    # "imputer__strategy": ('constant', 'mean', 'median', 'most_frequent'),
    # "imputer__fill_value": (0, -1),
    "clf__activation": ('identity', 'logistic', 'tanh', 'relu'),
    "clf__solver": ('lbfgs', 'sgd', 'adam'),
    "clf__hidden_layer_sizes": ((15,2), (100,), (15,15), (20,3), (20, 20, 20)),
    "clf__alpha": np.logspace(-8, 3, 12),
    "clf__max_iter": (200, 300),
}

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=parameter_grid,
    n_iter=50,
    random_state=1,
    n_jobs=4,
    verbose=1,
)

print("Performing grid search...")
print("Hyperparameters to be evaluated:")
pprint(parameter_grid)

from time import time

t0 = time()
search.fit(X_train,y_train)
print(f"Done in {time() - t0:.3f}s")

print("Best parameters combination found:")
best_parameters = search.best_estimator_.get_params()
for param_name in sorted(parameter_grid.keys()):
    print(f"{param_name}: {best_parameters[param_name]}")
    
clf = search.best_estimator_

test_accuracy = search.score(X_test, y_test)
print(
    "Accuracy of the best parameters using the inner CV of "
    # f"the random search: {random_search.best_score_:.3f}"
    f"the grid search: {search.best_score_:.3f}"
)
print(f"Accuracy on test set: {test_accuracy:.3f}")

# 3. classification
clf = MLPClassifier(
                    solver='adam',
                    # solver='sgd',
                    # alpha=1e-5,
                    # hidden_layer_sizes=(15, 2), 
                    random_state=1)

clf.fit(X_train, y_train)


# 4. performance evaluation
# accuracy & precision, false positives, false negatives

# X_train.shape, y_train.shape
# X_test.shape, y_test.shape

# clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    # hidden_layer_sizes=(15, 2), 
                    # random_state=1)


# clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
print(clf.score(X_test, y_test))
print("accurancy from holdout\n")

#crossvalidation
# clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    # hidden_layer_sizes=(15, 2), 
                    # random_state=1)
# scores = cross_val_score(clf, X, y, cv=10)
scores = cross_val_score(clf, X_train, y_train, cv=10)
print(scores)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


#some visulization?

