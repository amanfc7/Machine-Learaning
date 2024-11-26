#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
from ds_load_util import load_dataset

import sys

"""
    Call with flag '-s' to do a gridsearch (default = no search)
    Call with flag '--scaler' followed by a number to set the scaler used (default = MinMaxScaler)
"""
def main():
    search = True
    scaler_no = 2
    for i, arg in enumerate(sys.argv):
        if arg == '-s':
            search = True
        if arg == '--scaler':
            scaler_no = int(sys.argv[i+1])
    train_model(search,scaler_no)

"""
    Parameters:
        * scaler_no: 1 for preprocessing.StandardScaler(), 2 for preprocessing.MinMaxScaler(), 3 for preprocessing.RobustScaler(), else no scaler. Default=2
"""
# ----------------Wine-------------------------
def train_model(do_gridsearch=False, scaler_no=2, skip_eval=False):
    
    # 1. import data
    X_train, X_test, y_train, y_test  = load_dataset('wine', 
                                                     preprocess=True, 
                                                      scaler_no=scaler_no,
                                                 )
        
    # 2. gridsearch
    # Define different length_scale values to iterate over
    length_scales = [0.1, 1.0, 1.5, 2.0]

    # Create kernels with different length_scale values
    kernels = [
    RBF(length_scale=l) for l in length_scales
    ] + [
    Matern(length_scale=l) for l in length_scales
    ] + [
    RationalQuadratic(length_scale=l) for l in length_scales
    ]
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

        # 4. performance evaluation
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

   
