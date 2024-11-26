#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
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
    search = True
    for arg in sys.argv:
        if arg == '-s':
            search = True
    train_model(search)


# ----------------Congress-------------------------
def train_model(do_gridsearch=False, scaler_no=3, skip_eval=False):
    #TODO
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
    # Define different length_scale values to iterate over
    length_scales = [0.01, 0.1, 1.0, 1.5, 2.0, 2.5, 5, 10]

    # Create kernels with different length_scale values
    kernels = [
    RBF(length_scale=l) for l in length_scales
    ] + [
    Matern(length_scale=l) for l in length_scales
    ] + [
    RationalQuadratic(length_scale=l) for l in length_scales
    ]

    # Define the parameter grid for search
    parameters = {
        "kernel": kernels,  # Use the dynamically created kernels
        "optimizer": [None, "fmin_l_bfgs_b"],  # Optimizer options
        "max_iter_predict": [100, 300],  # Prediction iterations
        "n_restarts_optimizer": [0, 2],  # Restarts for optimizer
    }

         
    grid_search = GridSearchCV(
        GaussianProcessClassifier(random_state=1),
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
        clf = GaussianProcessClassifier(
            kernel=RBF(),               # RBF, Matern, RationalQuadratic
            optimizer="fmin_l_bfgs_b",
            max_iter_predict= 300,
            n_restarts_optimizer= 2,
        )

        clf.fit(X_train, y_train)

        # accuracy & precision, false positives, false negatives
        if not skip_eval:
            scores = cross_val_score(clf, X_train, y_train, cv=10)
            
            print(clf.score(X_test, y_test))
            print("accuracy from holdout\n")
            
            #crossvalidation
            print(scores)
            print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
        
    print("Scaler number: %d" % scaler_no)

    return (clf, X_test, y_test)


if __name__ == '__main__':
    main()
