#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct, ConstantKernel as C
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from pprint import pprint
from ds_load_util import load_dataset

import sys

def main():
    search = True
    scaler_no=1
    for arg in sys.argv:
        if arg == '-s':
            search = True
    train_model(search,scaler_no)

def train_model(do_search=False, scaler_no=1):
    # Select scaler  
    if scaler_no == 1:
        scaler = preprocessing.StandardScaler()
    elif scaler_no == 2:
        scaler=preprocessing.MinMaxScaler()
    elif scaler_no == 3:
        scaler=preprocessing.RobustScaler()
    elif scaler_no == 4:
        scaler = preprocessing.MaxAbsScaler()
    else:
        scaler = None
    
    # 1. import data
    X_train, X_test, y_train, y_test  = load_dataset('second2', 
                                                    #preprocess=True, 
                                                    #encoder=preprocessing.OrdinalEncoder(),
                                                    # #encoder=preprocessing.OneHotEncoder(),
                                                    # # imputer=SimpleImputer(strategy="constant", fill_value=-1),
                                                    #imputer=SimpleImputer(),
                                                    #scaler=scaler,
                                                    # # scaler= preprocessing.StandardScaler(with_mean=False),
                                                    #scaler= preprocessing.MaxAbsScaler(),
                                                    #("normalizer", preprocessing.Normalizer()),
                                                    )
    
    # # 2. data exploration and preprocessing
    categories_to_transform = ['sex', 'on_thyroxine', 'query_on_thyroxine',
                                  'on_antithyroid_medication','sick','pregnant',
                                  'thyroid_surgery','I131_treatment','query_hypothyroid',
                                  'query_hyperthyroid','lithium','goitre',
                                       'tumor','hypopituitary','psych','TSH_measured',
                                  'T3_measured','TT4_measured','T4U_measured',
                                  'FTI_measured','referral_source']
   
    enc = preprocessing.OrdinalEncoder()
   
    X_train[categories_to_transform] = enc.fit_transform(X_train[categories_to_transform])
    X_test[categories_to_transform] = enc.transform(X_test[categories_to_transform])
   
    enc = Pipeline(steps=[
        # ("encoder", preprocessing.OrdinalEncoder()),
        # ("encoder", preprocessing.OneHotEncoder()),
        ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
        ("scaler", scaler),
        # ("scaler", preprocessing.StandardScaler(with_mean=False)),
        # ("normalizer", preprocessing.Normalizer()),
    ])
    
    X_train = enc.fit_transform(X_train)
    X_test = enc.transform(X_test)
    
    #NOTE: could have gridsearch over different encoders etc via lambda function or similar
    
    kernels = [RBF(length_scale=0.1), Matern(length_scale=0.1), RationalQuadratic(length_scale=0.1)]


    # 2. gridsearch
    # parameter tuning
    if do_search:
        parameters = {
            "kernel": kernels,
            "optimizer": [None, "fmin_l_bfgs_b"],
            "max_iter_predict": [100, 300],
            #"length_scale" : [1.5, 2.0],
            #"length_bond_scales" : [(1e-2, 1e2), (1e-3, 1e3)],
            "n_restarts_optimizer": [0, 2],
        }

        search = RandomizedSearchCV(
            estimator=GaussianProcessClassifier(random_state=1),
            param_distributions=parameters,
            # parameter_grid,
            n_iter=60,
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
        
        # Print the results of all the parameter combinations
        print("\nResults of each parameter combination:")
        results = search.cv_results_
        for i in range(len(results['mean_test_score'])):
            # Extracting the parameters for each combination
            kernel = results['params'][i].get('kernel', 'N/A')
            optimizer = results['params'][i].get('optimizer', 'N/A')
            accuracy = results['mean_test_score'][i]
            print(f"Scaler: {scaler_no}, Kernel: {kernel}, Optimizer: {optimizer}, Accuracy: {accuracy:.3f}")
    
    else:
        if scaler_no == 1: #StandardScaler
            clf = GaussianProcessClassifier()
        elif scaler_no == 2:# MinMaxScaler
            clf = GaussianProcessClassifier()
        elif scaler_no == 3:#RobustScaler
            clf = GaussianProcessClassifier()
        else: #no scaler
            clf = GaussianProcessClassifier()
            
        
        
        clf.fit(X_train, y_train)
        
        # accuracy & precision, false positives, false negatives
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
