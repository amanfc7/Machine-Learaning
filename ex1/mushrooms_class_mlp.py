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
from pprint import pprint
from ds_load_util import load_dataset

import sys

def main():
    search = False
    for arg in sys.argv:
        if arg == '-s':
            search = True
    train_model(search)

def train_model(do_gridsearch=False):
    # ----------------Mushrooms-------------------------
    
    # 1. import data
    X_train, X_test, y_train, y_test  = load_dataset('mushroom', 
                                                      preprocess=True, 
                                                      encoder=preprocessing.OrdinalEncoder(),
                                                      # encoder=preprocessing.OneHotEncoder(),
                                                       imputer=SimpleImputer(strategy="constant", fill_value=-1),
                                                      # imputer=SimpleImputer(),
                                                       scaler= preprocessing.StandardScaler(),
                                                      # scaler= preprocessing.StandardScaler(with_mean=False),
                                                      # ("scaler", preprocessing.RobustScaler()),
                                                      # scaler= preprocessing.MaxAbsScaler(),
                                                      # scaler= preprocessing.RobustScaler(),
                                                      # scaler= preprocessing.MinMaxScaler(),
                                                      # ("normalizer", preprocessing.Normalizer()),
                                                      )
    
    
    # # 2. data exploration and preprocessing
    # enc = Pipeline(steps=[
    #     # ("o_encoder", preprocessing.OrdinalEncoder()),
    #     # ("oh_encoder", preprocessing.OneHotEncoder()),
    #     ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
    #     ("scaler", preprocessing.StandardScaler()),
    #     # ("scaler", preprocessing.RobustScaler()),
    #     # ("scaler", preprocessing.MaxAbsScaler()),
    #     # ("scaler", preprocessing.MinMaxScaler()),
    #     # ("scaler", preprocessing.StandardScaler(with_mean=False)),
    #     # ("normalizer", preprocessing.Normalizer()),
    # ])
    # X_train = enc.fit_transform(X_train)
    # X_test = enc.transform(X_test)
    
    if do_gridsearch:
    
        # parameter tuning
        parameters = {
            "activation": ('identity', 'logistic', 'tanh', 'relu'),
            "solver": ('lbfgs', 'sgd', 'adam'),
            "hidden_layer_sizes": ((15,2), (100,), (15,15), (20,3), (50, 50), (20, 20, 20)),
            "alpha": np.logspace(-10, 4, 15),
            # "max_iter": (200, 300),
        }

        search = RandomizedSearchCV(
            estimator=MLPClassifier(random_state=1),
            param_distributions=parameters,
            # parameter_grid,
            n_iter=50,
            random_state=1,
            n_jobs=4,
            verbose=1,
        )

        print("Performing grid search...")
        print("Hyperparameters to be evaluated:")
        # pprint(parameter_grid)

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
        # 3. classification
        # clf = MLPClassifier(solver='lbfgs', 
        #                     alpha=1e-5,
        #                     hidden_layer_sizes=(15, 2), 
        #                     # activation='relu',
        #                     random_state=1)
        clf = MLPClassifier(solver='adam', 
                            alpha=1e-10,
                            hidden_layer_sizes=(15, 2), 
                            activation='relu',
                            random_state=1)
        
        clf.fit(X_train, y_train)
        
        
        # 4. performance evaluation
        # accuracy & precision, false positives, false negatives
        
        scores = cross_val_score(clf, X_train, y_train, cv=10)
        
        
        print(clf.score(X_test, y_test))
        print("accuracy from holdout\n")
        
        #crossvalidation
        print(scores)
        print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    
    
    # #some visulization?
    


if __name__ == '__main__':
    main()