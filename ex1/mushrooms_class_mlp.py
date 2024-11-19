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
from pprint import pprint

# ----------------Mushrooms-------------------------
do_gridsearch = True

# 1. import data
ds = fetch_openml(name='mushroom', version=1)
X = ds.data
X = X.where(X=='?', other=np.nan)
y = ds.target

# 1.1. split data for holdout
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)
    # X_preprocessed, y, test_size=0.4, random_state=0)

# 2. data exploration and preprocessing
enc = Pipeline(steps=[
    ("o_encoder", preprocessing.OrdinalEncoder()),
    # ("oh_encoder", preprocessing.OneHotEncoder()),
    ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
    ("scaler", preprocessing.StandardScaler()),
    # ("scaler", preprocessing.RobustScaler()),
    # ("scaler", preprocessing.MaxAbsScaler()),
    # ("scaler", preprocessing.MinMaxScaler()),
    # ("scaler", preprocessing.StandardScaler(with_mean=False)),
    # ("normalizer", preprocessing.Normalizer()),
])
X_train = enc.fit_transform(X_train)
X_test = enc.transform(X_test)


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
# clf = MLPClassifier(solver='lbfgs', 
#                     alpha=1e-5,
#                     hidden_layer_sizes=(15, 2), 
#                     # activation='relu',
#                     random_state=1)
clf = MLPClassifier(solver='lbfgs', 
                    alpha=1e-3,
                    hidden_layer_sizes=(15, 2), 
                    activation='logistic',
                    random_state=1)

clf.fit(X_train, y_train)


# 4. performance evaluation
# accuracy & precision, false positives, false negatives


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


# #some visulization?


    
# # 2. preprocessing & gridsearch
# pipeline = Pipeline([
#     ("o_encoder", preprocessing.OrdinalEncoder()), #NO CATEGORICAL DATA
#     # ("oh_encoder", preprocessing.OneHotEncoder()),
#     ("imputer", SimpleImputer()),
#     ("scaler", preprocessing.StandardScaler()),
#     # ("scaler", preprocessing.RobustScaler()),
#     # ("scaler", preprocessing.MaxAbsScaler()),
#     # ("scaler", preprocessing.MinMaxScaler()),
#     # ("scaler", preprocessing.StandardScaler(with_mean=False)),
#     # ("normalizer", preprocessing.Normalizer()),
#     ("clf", MLPClassifier(random_state=1)),
# ])

# parameter_grid = {
#     # "oh_encoder__drop": (None, "if_binary"),
#     # # "oh_encoder__handle_unknown": ("infrequent_if_exist", "error"),
#     # "imputer__strategy": ('constant', 'mean', 'median', 'most_frequent'),
#     # "imputer__fill_value": (0, -1),
#     # "scaler__with_mean": (True, False),
#     "clf__activation": ('identity', 'logistic', 'tanh', 'relu'),
#     "clf__solver": ('lbfgs', 'sgd', 'adam'),
#     "clf__hidden_layer_sizes": ((15,2), (100,), (15,15), (20,3)),
#     "clf__alpha": np.logspace(-8, 3, 12),
#     "clf__max_iter": (200, 300),
# }

# # random_search = RandomizedSearchCV(
# #     estimator=pipeline,
# #     param_distributions=parameter_grid,
# #     n_iter=40,
# #     random_state=0,
# #     n_jobs=2,
# #     verbose=1,
# # )
# grid_search = GridSearchCV(
#     estimator=pipeline,
#     param_grid=parameter_grid,
#     # n_iter=40,
#     n_jobs=2,
#     verbose=1,
# )

# if do_gridsearch:
#     print("Performing grid search...")
#     print("Hyperparameters to be evaluated:")
#     # pprint(parameter_grid)
    
#     from time import time
    
#     t0 = time()
#     # random_search.fit(X_train,y_train)
#     grid_search.fit(X_train, y_train)
#     print(f"Done in {time() - t0:.3f}s")
    
#     print("Best parameters combination found:")
#     # best_parameters = random_search.best_estimator_.get_params()
#     best_parameters = grid_search.best_estimator_.get_params()
#     for param_name in sorted(parameter_grid.keys()):
#         print(f"{param_name}: {best_parameters[param_name]}")
    
#     clf = grid_search
    
#     test_accuracy = grid_search.score(X_test, y_test)
#     print(
#         "Accuracy of the best parameters using the inner CV of "
#         # f"the random search: {random_search.best_score_:.3f}"
#         f"the grid search: {grid_search.best_score_:.3f}"
#     )
#     print(f"Accuracy on test set: {test_accuracy:.3f}")

# else:
#     # fitting
#     pipeline_adjusted = Pipeline([
#         # ("o_encoder", preprocessing.OrdinalEncoder()),
#         # ("oh_encoder", preprocessing.OneHotEncoder()),
#         # ("imputer", SimpleImputer()),
#         # ("imputer", SimpleImputer(strategy='constant',fill_value=-1)),
#         ("scaler", preprocessing.StandardScaler()),
#         # ("scaler", preprocessing.RobustScaler()),
#         # ("scaler", preprocessing.MaxAbsScaler()),
#         # ("scaler", preprocessing.MinMaxScaler()),
#         # ("scaler", preprocessing.StandardScaler(with_mean=False)),
#         # ("normalizer", preprocessing.Normalizer()),
#         ("clf", MLPClassifier(solver='lbfgs', 
#                           alpha=1e1,
#                           activation='identity',
#                           hidden_layer_sizes=(15, 15), 
#                           random_state=1)),
#     ])
#     pipeline_adjusted.fit(X_train,y_train)
#     clf = pipeline_adjusted



#     # 4. performance evaluation
#     # accuracy & precision, false positives, false negatives
    
#     print(clf.score(X_test, y_test))
#     print("accurancy from holdout\n")
    
#     #crossvalidation
#     # scores = cross_val_score(clf, X, y, cv=10)
#     # if not grid_search:
#     scores = cross_val_score(clf, X_train, y_train, cv=10)
#     print(scores)
#     print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

