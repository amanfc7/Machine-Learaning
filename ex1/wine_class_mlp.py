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
# from pprint import pprint
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
    parameters = {
        "activation": ('identity', 'logistic', 'tanh', 'relu'),
        "solver": ('lbfgs', 'sgd', 'adam'),
        "hidden_layer_sizes": ((15,2), (100,), (15,15), (20,3)),
        "alpha": np.logspace(-8, 3, 12),
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
        if scaler_no == 1:
            clf = MLPClassifier(solver='lbfgs', #better w/ StandardScaler
                                  alpha=1e1,
                                  activation='identity',
                                  hidden_layer_sizes=(15, 15), 
                                  random_state=1)
        if scaler_no == 2:
            clf = MLPClassifier(solver='lbfgs', #better w/MinMaxScaler
                                alpha=1e-4,
                                hidden_layer_sizes=(15, 2), 
                                activation='logistic',
                                random_state=1)
        if scaler_no == 3:
            clf = MLPClassifier(solver='sgd', #better w/RobustScaler
                                alpha=1e-8,
                                hidden_layer_sizes=(100,), 
                                activation='identity',
                                random_state=1)
        
        clf.fit(X_train, y_train)
    
    
    
        # 4. performance evaluation
        # accuracy & precision, recall
        if not skip_eval:
            scores = cross_val_score(clf, X_train, y_train, cv=10)
            
            print(clf.score(X_test, y_test))
            print("accuracy from holdout\n")
            
            average = 'macro'
            print(precision_score(y_test, clf.predict(X_test), average=average))
            print("precision from holdout\n")
            
            print(recall_score(y_test, clf.predict(X_test), average= average))
            print("recall from holdout\n")
            
            #crossvalidation
            print(scores)
            print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
        
    print("Scaler number: %d" % scaler_no)

    return (clf, X_test, y_test)
        

if __name__ == '__main__':
    main()