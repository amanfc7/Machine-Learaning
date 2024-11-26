#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from pprint import pprint
from ds_load_util import load_dataset

import sys

"""
    Call with flag '-s' to do a gridsearch (default = no search)
    Call with flag '--scaler' followed by a number to set the scaler used (default = StandardScaler)
"""
def main():
    search = False
    scaler_no = 1
    for i, arg in enumerate(sys.argv):
        if arg == '-s':
            search = True
        if arg == '--scaler':
            scaler_no = int(sys.argv[i+1])
    train_model(search,scaler_no)

    # ----------------2nd own data set-------------------------
"""
    Parameters:
        * scaler_no: 1 for preprocessing.StandardScaler(), 2 for preprocessing.MinMaxScaler(), 3 for preprocessing.RobustScaler(), 4 for preprocessing.MaxAbsScaler(), else no scaler. Default=1
"""
def train_model(do_search=False, scaler_no=1, skip_eval=False):

    
    # 1. import data
    X_train, X_test, y_train, y_test  = load_dataset('second2', 
                                                       preprocess=True, 
                                                        encoder=preprocessing.OrdinalEncoder(),
                                                     # # encoder=preprocessing.OneHotEncoder(),
                                                      imputer=SimpleImputer(strategy="constant", fill_value=-1),
                                                        # imputer=SimpleImputer(),
                                                        scaler_no=scaler_no,
                                                     # # scaler= preprocessing.StandardScaler(with_mean=False),
                                                     # # ("normalizer", preprocessing.Normalizer()),
                                                     )
    
    # # 2. gridsearch
    # parameters = {
    #     "activation": ('identity', 'logistic', 'tanh', 'relu'),
    #     "solver": ('lbfgs', 'sgd', 'adam'),
    #     "hidden_layer_sizes": ((15,2), (100,), (15,15), (20,3), (50, 50), (20, 20, 20)),
    #     "alpha": np.logspace(-10, 4, 15),
    #     # "max_iter": (200, 300),
    # }
    
    # grid_search = GridSearchCV(
    #     MLPClassifier(random_state=1),
    #     parameters,
    #     # n_iter=40,
    #     n_jobs=4,
    #     # verbose=1,
    # )
    
    # if do_gridsearch:
    #     print("Performing grid search...")
    #     print("Hyperparameters to be evaluated:")
    #     # pprint(parameter_grid)
        
    #     from time import time
        
    #     t0 = time()
    #     grid_search.fit(X_train, y_train)
    #     print(f"Done in {time() - t0:.3f}s")
        
    #     print("Best parameters combination found:")
    #     best_parameters = grid_search.best_estimator_.get_params()
    #     for param_name in sorted(parameters.keys()):
    #         print(f"{param_name}: {best_parameters[param_name]}")
        
    #     clf = grid_search.best_estimator_
        
    #     test_accuracy = grid_search.score(X_test, y_test)
    #     print(
    #         "Accuracy of the best parameters using the inner CV of "
    #         f"the grid search: {grid_search.best_score_:.3f}"
    #     )
    #     print(f"Accuracy on test set: {test_accuracy:.3f}")
    
    # else:
    # if scaler_no == 1: #better w/ StandardScaler
    clf = DecisionTreeClassifier()
    # elif scaler_no == 2:  #better w/MinMaxScaler TODO
    #     clf = MLPClassifier(solver='lbfgs',
    #                     alpha=1e-4,
    #                     hidden_layer_sizes=(15, 2), 
    #                     activation='logistic',
    #                     random_state=1)
    # elif scaler_no == 3: #better w/RobustScaler
    #     clf = MLPClassifier(solver='lbfgs', 
    #                     alpha=1e-0,
    #                     hidden_layer_sizes=(15,2), 
    #                     activation='identity',
    #                     random_state=1)
    
    clf.fit(X_train, y_train)
    
    # accuracy & precision, false positives, false negatives
    if not skip_eval:
        scores = cross_val_score(clf, X_train, y_train, cv=10)
    
        print(clf.score(X_test, y_test))
        print("accurancy from holdout\n")
    
        #crossvalidation
        # clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                            # hidden_layer_sizes=(15, 2), 
                            # random_state=1)
        # scores = cross_val_score(clf, X, y, cv=10)
        print(scores)
        print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    
    #some visulization?
    print("Scaler number: %d" % scaler_no)
    
    return (clf, X_test, y_test)


if __name__ == '__main__':
    main()
