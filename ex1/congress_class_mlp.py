#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import numpy as np

# from pprint import pprint
import pandas as pd
import matplotlib.pyplot as plt

from ds_load_util import load_dataset
# import seaborn as sns
# %matplotlib inline

import sys

def main():
    search = False
    for arg in sys.argv:
        if arg == '-s':
            search = True
    train_model(search)

# ----------------Congress-------------------------
def train_model(do_gridsearch=False, scaler_no=3):
    
    if scaler_no == 1:
        scaler = preprocessing.StandardScaler()
    if scaler_no == 11:
        scaler = preprocessing.StandardScaler(with_mean=False)
    elif scaler_no == 2:
        scaler=preprocessing.MinMaxScaler()
    elif scaler_no == 3:
        scaler=preprocessing.RobustScaler()
    elif scaler_no == 4:
        scaler=preprocessing.MaxAbsScaler()
    else:
        scaler = None
    
    # 1. import data
    X_train, X_test, y_train, y_test  = load_dataset('congress', 
                                                     preprocess=True, 
                                                      encoder=preprocessing.OrdinalEncoder(),
                                                     # encoder=preprocessing.OneHotEncoder(),
                                                      # imputer=SimpleImputer(strategy="constant", fill_value=-1),
                                                      imputer=SimpleImputer(),
                                                      scaler=scaler,
                                                     # ("normalizer", preprocessing.Normalizer()),
                                                     )
    
    # # 2. data exploration and preprocessing
    # enc = Pipeline(steps=[
    #     ("encoder", preprocessing.OrdinalEncoder()),
    #     # ("encoder", preprocessing.OneHotEncoder()),
    #     ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
    #     ("scaler", preprocessing.StandardScaler()),
    #     # ("scaler", preprocessing.RobustScaler()),
    #     # ("scaler", preprocessing.MaxAbsScaler()),
    #     # ("scaler", preprocessing.MinMaxScaler()),
    #     # ("scaler", preprocessing.StandardScaler(with_mean=False)),
    #     # ("normalizer", preprocessing.Normalizer()),
    # ])
    
    # 2. gridsearch
    parameters = {
        "activation": ('identity', 'logistic', 'tanh', 'relu'),
        "solver": ('lbfgs', 'sgd', 'adam'),
        "hidden_layer_sizes": ((15,2), (100,), (15,15), (20,3), (50, 50), (20, 20, 20)),
        "alpha": np.logspace(-10, 4, 15),
        # "max_iter": (200, 300),
    }
    
    grid_search = GridSearchCV(
        MLPClassifier(random_state=1),
        parameters,
        # n_iter=40,
        n_jobs=4,
        # verbose=1,
    )
    
    if do_gridsearch:
        print("Performing grid search...")
        print("Hyperparameters to be evaluated:")
        # pprint(parameter_grid)
        
        from time import time
        
        t0 = time()
        grid_search.fit(X_train, y_train)
        print(f"Done in {time() - t0:.3f}s")
        
        print("Best parameters combination found:")
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print(f"{param_name}: {best_parameters[param_name]}")
        
        clf = grid_search.best_estimator_
        
        test_accuracy = grid_search.score(X_test, y_test)
        print(
            "Accuracy of the best parameters using the inner CV of "
            f"the grid search: {grid_search.best_score_:.3f}"
        )
        print(f"Accuracy on test set: {test_accuracy:.3f}")
    
    else:
        if scaler_no == 1: #better w/ StandardScaler
            clf = MLPClassifier(solver='lbfgs', 
                                  alpha=1e-10,
                                  activation='logistic',
                                  hidden_layer_sizes=(20, 20, 20), 
                                  random_state=1)
        elif scaler_no == 2:  #better w/MinMaxScaler TODO
            clf = MLPClassifier(solver='lbfgs',
                            alpha=1e-4,
                            hidden_layer_sizes=(15, 2), 
                            activation='logistic',
                            random_state=1)
        elif scaler_no == 3: #better w/RobustScaler
            clf = MLPClassifier(solver='lbfgs', 
                            alpha=1e-0,
                            hidden_layer_sizes=(15,2), 
                            activation='identity',
                            random_state=1)
        
        clf.fit(X_train, y_train)
        
        # accuracy & precision, false positives, false negatives
        
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

    return clf


if __name__ == '__main__':
    main()