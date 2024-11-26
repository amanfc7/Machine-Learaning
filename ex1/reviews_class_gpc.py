#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pprint import pprint
from ds_load_util import load_dataset
import sys

"""
    Call with flag '-s' to do a gridsearch (default = no search)
    Call with flag '--scaler' followed by a number to set the scaler used (default = MinMaxScaler)
"""
def main():
    search = False
    scaler_no = 2
    for i, arg in enumerate(sys.argv):
        if arg == '-s':
            search = True
        if arg == '--scaler':
            scaler_no = int(sys.argv[i+1])
    train_model(search, scaler_no)

"""
    Parameters:
        * scaler_no: 1 for preprocessing.StandardScaler(), 2 for preprocessing.MinMaxScaler(), 3 for preprocessing.RobustScaler(), else no scaler. Default=2
"""
def train_model(do_search=False, scaler_no=2, skip_eval=False):
    # ----------------Reviews-------------------------
    
    # 1. import data
    X_train, X_test, y_train, y_test  = load_dataset('reviews', 
                                                      preprocess=True, 
                                                     #  encoder=preprocessing.OrdinalEncoder(),
                                                     # # encoder=preprocessing.OneHotEncoder(),
                                                     #  # imputer=SimpleImputer(strategy="constant", fill_value=-1),
                                                     #  imputer=SimpleImputer(),
                                                       scaler_no=scaler_no,
                                                     # # scaler= preprocessing.StandardScaler(with_mean=False),
                                                     #  scaler= preprocessing.MaxAbsScaler(),
                                                     # # ("normalizer", preprocessing.Normalizer()),
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
    
    #NOTE: could have gridsearch over different encoders etc via lambda function or similar
    
    # 2. gridsearch
    # parameter tuning
    if do_search:
        length_scales = [1.0, 1.5, 2.0]
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

        search = RandomizedSearchCV(
            estimator=GaussianProcessClassifier(random_state=1),
            param_distributions=parameters,
            # parameter_grid,
            n_iter=60,
            cv=5,
            random_state=1,
            n_jobs=4,
            verbose=1,
        )

        print("Performing grid search...")
        print("Hyperparameters to be evaluated:")
        from time import time
        t0 = time()
        search.fit(X_train,y_train)
        print(f"Done in {time() - t0:.3f}s")
        print("Best parameters combination found:")
        best_parameters = search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print(f"{param_name}: {best_parameters[param_name]}")
            
        clf = search.best_estimator_
        test_accuracy = search.score(X_test, y_test)
        print(
            "Accuracy of the best parameters using the inner CV of "
            # f"the random search: {random_search.best_score_:.3f}"
            f"the grid search: {search.best_score_:.3f}"
        )
        print(f"Accuracy on test set: {test_accuracy:.3f}")

    else:
        clf = GaussianProcessClassifier(
            kernel=RationalQuadratic(length_scale=1.5),               # RBF, Matern, RationalQuadratic
            optimizer="fmin_l_bfgs_b",
            max_iter_predict= 300,
            n_restarts_optimizer= 2,
        )


        clf.fit(X_train, y_train)
        
        # accuracy & precision, false positives, false negatives
        if not skip_eval:
            scores = cross_val_score(clf, X_train, y_train, cv=10)
            print(clf.score(X_test, y_test))
            print("accurancy from holdout\n")
            #crossvalidation
            print(scores)
            print("CV: %0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    
    #some visulization?
    
    print("Scaler number: %d" % scaler_no)

    return (clf, X_test, y_test)

if __name__ == '__main__':
    main()
