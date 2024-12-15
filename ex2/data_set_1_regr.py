#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time

from random_forest_regressor import RandomForestRegressor
from DT_regressor import DTRegressor


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import pandas as pd


def train_model():
    ds = pd.read_csv('colleges_data_preprocessed.csv')
    # if verbose:
    #     print(ds.sample(3))
    X = ds.drop(['UNITID','percent_pell_grant'], axis=1)
    # X = X.where(X!='unknown', other=np.nan)
    y = ds['percent_pell_grant']
    
    # print(X)
    # print(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)
    
    t0= time.time()
    clf = RandomForestRegressor(use_skl_tree=False, max_samples=100, max_features=20)
    # clf = RandomForestRegressor()
    clf.fit(X_train, y_train)
    print("Training time: %f" % (time.time() - t0))
    
    # y_prediction = clf.predict(X_test)

    
    # t0= time.time()
    # clf = DTRegressor(splitter='random')
    # clf.fit(X_train, y_train) 
    # print(time.time() - t0)
    
    
    #some quick evaluation
    # print(r2_score(y_test, clf_t_skl.predict(X_test)))
    t0= time.time()
    y_pred_test = clf.predict(X_test)
    print("Prediction time: %f" % (time.time() - t0))

    print("R2 score: %f" % r2_score(y_test, y_pred_test ))
    print("MSE: %f" % mean_squared_error(y_test, y_pred_test ))
    print("MAE: %f" % mean_absolute_error(y_test, y_pred_test ))
    

def main():
    train_model()
    



if __name__ == '__main__':
    main()