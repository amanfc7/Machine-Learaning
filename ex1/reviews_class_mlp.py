#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score
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
        parameters = {
            "activation": ('identity', 'logistic', 'tanh', 'relu'),
            "solver": ('lbfgs', 'sgd', 'adam'),
            "hidden_layer_sizes": ((15,2), (100,), (15,15), (20,3), (50, 50), (20, 20, 20), (100,100,100)),
            "alpha": np.logspace(-10, 4, 15),
            # "max_iter": (200, 300),
        }
        search = RandomizedSearchCV(
            estimator=MLPClassifier(random_state=1),
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
        
    
    else:
        if scaler_no == 1: #StandardScaler
            clf = MLPClassifier(solver='adam',
                                alpha=1e-1,
                                hidden_layer_sizes=(100,), 
                                activation='tanh',
                                random_state=1)
        elif scaler_no == 2:# MinMaxScaler
            clf = MLPClassifier(solver='adam',
                                  alpha=1e-1,
                                  activation='tanh',
                                  hidden_layer_sizes=(100,), 
                                  random_state=1)
        elif scaler_no == 3:#RobustScaler
            clf = MLPClassifier(solver='lbfgs',
                                alpha=1e1,
                                hidden_layer_sizes=(100,), 
                                activation='relu',
                                random_state=1)
        
        else: #no scaler
            clf = MLPClassifier(solver='lbfgs',
                                alpha=1e1,
                                hidden_layer_sizes=(100,), 
                                activation='identity',
                                random_state=1)
            
        
        
        clf.fit(X_train, y_train)
        
        # accuracy & precision, false positives, false negatives
        if not skip_eval:
            scores = cross_val_score(clf, X_train, y_train, cv=10)
            print(clf.score(X_test, y_test))
            print("accurancy from holdout\n")
            
            average = 'macro'
            print(precision_score(y_test, clf.predict(X_test), average=average))
            print("precision from holdout\n")
            
            print(recall_score(y_test, clf.predict(X_test), average= average))
            print("recall from holdout\n")
            
            #crossvalidation
            print(scores)
            print("CV: %0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    
    #some visulization?
    
    print("Scaler number: %d" % scaler_no)

    return (clf, X_test, y_test)

if __name__ == '__main__':
    main()