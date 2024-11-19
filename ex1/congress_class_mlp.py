# from ucimlrepo import fetch_ucirepo 
# import matplotlib

# import arff, numpy as np

from sklearn.datasets import fetch_openml, load_wine
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import numpy as np

from pprint import pprint
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

def train_model(do_gridsearch=False):
    # ----------------Congress-------------------------
    
    
    # 1. import data
    X_train, X_test, y_train, y_test  = load_dataset('congress', 
                                                     preprocess=True, 
                                                      encoder=preprocessing.OrdinalEncoder(),
                                                     # encoder=preprocessing.OneHotEncoder(),
                                                      imputer=SimpleImputer(strategy="constant", fill_value=-1),
                                                     # imputer=SimpleImputer(),
                                                     # scaler= preprocessing.StandardScaler(),
                                                     # scaler= preprocessing.StandardScaler(with_mean=False),
                                                     # ("scaler", preprocessing.RobustScaler()),
                                                      scaler= preprocessing.MaxAbsScaler(),
                                                     # scaler= preprocessing.MinMaxScaler(),
                                                     # ("normalizer", preprocessing.Normalizer()),
                                                     )
    # ds = pd.read_csv('CongressionalVotingID.shuf.lrn.csv')
    # # ds = fetch_openml(name='mushroom', version=1)
    # # print(ds)
    # print(ds.sample(3))
    # X = ds.drop(['ID', 'class'], axis=1)
    # X = X.where(X=='unknown', other=np.nan)
    # y = ds['class']
    
    # # 1.1. split data for holdout
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.3, random_state=0)
    #     # X_preprocessed, y, test_size=0.4, random_state=0)
    
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
    # X_train = enc.fit_transform(X_train)
    # X_test = enc.transform(X_test)
    
    # # 2. data exploration and preprocessing
    # enc = Pipeline(steps=[
    #     ("encoder", preprocessing.OrdinalEncoder()),
    #     # ("encoder", preprocessing.OneHotEncoder()),
    #     ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
    #     ("scaler", preprocessing.StandardScaler()),
    # ])
    # X_preprocessed = enc.fit_transform(X)
    # # scaler = preprocessing.StandardScaler().fit(X)
    # # X_scaled = scaler.transform(X)
    # X_scaled = X_preprocessed
    
    
    # 3. classification
    if do_gridsearch:
        # parameter tuning
        pipeline = Pipeline([
            # ("o_encoder", preprocessing.OrdinalEncoder()),
            # # ("oh_encoder", preprocessing.OneHotEncoder()),
            # # ("imputer", SimpleImputer()),
            # ("imputer", SimpleImputer(strategy='constant',fill_value=-1)),
            # ("scaler", preprocessing.StandardScaler()),
            # # ("scaler", preprocessing.RobustScaler()),
            # # ("scaler", preprocessing.MaxAbsScaler()),
            # # ("scaler", preprocessing.MinMaxScaler()),
            # # ("scaler", preprocessing.StandardScaler(with_mean=False)),
            ("clf", MLPClassifier(random_state=1)),
        ])
        
        parameters = {
            "activation": ('identity', 'logistic', 'tanh', 'relu'),
            "solver": ('lbfgs', 'sgd', 'adam'),
            "hidden_layer_sizes": ((15,2), (100,), (15,15), (20,3), (50, 50), (20, 20, 20)),
            "alpha": np.logspace(-10, 4, 15),
            # "max_iter": (200, 300),
        }
        
        # parameter_grid = {
        #     # "oh_encoder__drop": (None, "if_binary"),
        #     # "oh_encoder__handle_unknown": ("infrequent_if_exist"),
        #     # "imputer__strategy": ('constant', 'mean', 'median', 'most_frequent'),
        #     # "imputer__fill_value": (0, -1),
        #     "clf__activation": ('identity', 'logistic', 'tanh', 'relu'),
        #     "clf__solver": ('lbfgs', 'sgd', 'adam'),
        #     "clf__hidden_layer_sizes": ((15,2), (100,), (15,15), (20,3), (20, 20, 20)),
        #     "clf__alpha": np.logspace(-8, 3, 12),
        #     "clf__max_iter": (200, 300),
        # }
        search = GridSearchCV(
            MLPClassifier(random_state=1),
            parameters,
            # n_iter=40,
            n_jobs=4,
            # verbose=1,
        )
        # search = RandomizedSearchCV(
        #     estimator=pipeline,
        #     param_distributions=parameters,
        #     n_iter=50,
        #     random_state=1,
        #     n_jobs=4,
        #     verbose=1,
        # )
        
        print("Performing grid search...")
        print("Hyperparameters to be evaluated:")
        pprint(parameters)
        
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
        # clf = MLPClassifier(solver='lbfgs', 
        #                     alpha=1e-3,
        #                     hidden_layer_sizes=(15, 2), 
        #                     activation='logistic',
        #                     random_state=1)
        clf = MLPClassifier(solver='adam', 
                            alpha=1e-8,
                            hidden_layer_sizes=(50, 50), 
                            activation='relu',
                            random_state=1)
        
        clf.fit(X_train, y_train)
    
    # 4. performance evaluation
    # accuracy & precision, false positives, false negatives
    
    print(clf.score(X_test, y_test))
    print("accurancy from holdout\n")
    
    #crossvalidation
    # clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        # hidden_layer_sizes=(15, 2), 
                        # random_state=1)
    # scores = cross_val_score(clf, X, y, cv=10)
    scores = cross_val_score(clf, X_train, y_train, cv=10)
    print(scores)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    
    
    #some visulization?


if __name__ == '__main__':
    main()