#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from pprint import pprint
from ds_load_util import load_dataset

import sys

def main():
    search = False
    for arg in sys.argv:
        if arg == '-s':
            search = True
    train_model(search)

def train_model(do_search=False):
    # ----------------2nd own data set-------------------------
    
    scaler_no = 1
    
    
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
    X_train, X_test, y_train, y_test  = load_dataset('second', 
                                                       preprocess=True, 
                                                       # encoder=preprocessing.OrdinalEncoder(),
                                                     # # encoder=preprocessing.OneHotEncoder(),
                                                     #  # imputer=SimpleImputer(strategy="constant", fill_value=-1),
                                                        imputer=SimpleImputer(),
                                                        scaler=scaler,
                                                     # # scaler= preprocessing.StandardScaler(with_mean=False),
                                                     #  scaler= preprocessing.MaxAbsScaler(),
                                                     # # ("normalizer", preprocessing.Normalizer()),
                                                     )
    
    # # # 2. data exploration and preprocessing
    # categories_to_transform = ['workclass','education','marital-status',
    #                            'occupation','relationship','race', 'sex', 
    #                            'native-country']
    # enc = preprocessing.OrdinalEncoder()
    # X_train[categories_to_transform] = enc.fit_transform(X_train[categories_to_transform])
    # X_test[categories_to_transform] = enc.transform(X_test[categories_to_transform])
    # enc = Pipeline(steps=[
    #     # ("encoder", preprocessing.OrdinalEncoder()),
    #     # ("encoder", preprocessing.OneHotEncoder()),
    #     ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
    #     ("scaler", scaler),
    #     # ("scaler", preprocessing.StandardScaler(with_mean=False)),
    #     # ("normalizer", preprocessing.Normalizer()),
    # ])
    # X_train = enc.fit_transform(X_train)
    # X_test = enc.transform(X_test)
    
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
        
        scores = cross_val_score(clf, X_train, y_train, cv=10)

        print(clf.score(X_test, y_test))
        print("accurancy from holdout\n")

        #crossvalidation
        print(scores)
        print("CV: %0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    
    #some visulization?
    
    print("Scaler number: %d" % scaler_no)


if __name__ == '__main__':
    main()
