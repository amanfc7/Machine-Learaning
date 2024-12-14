#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import time

from random_forest_regressor import RandomForestRegressor
from DT_regressor import DTRegressor

from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import pandas as pd
from ucimlrepo import fetch_ucirepo 

def train_model():
    ds = fetch_ucirepo(id=477) 
    X = ds.data.features 
    y = ds.data.targets 
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)
    
    t0= time.time()
    # clf = RandomForestRegressor(use_skl_tree=False, max_samples=100, max_features=20)
    clf = RandomForestRegressor()
    # clf = RandomForestRegressor(use_skl_tree=True)
    # clf = RandomForestRegressor(criterion="absolute_error")
    # clf = RFR(criterion="absolute_error")
    # clf = RFR()
    clf.fit(X_train, y_train)
    print("Training time: %f" % (time.time() - t0))
    
    # y_prediction = clf.predict(X_test)

    
    # t0= time.time()
    # clf = DTRegressor(splitter='random')
    # clf.fit(X_train, y_train) 
    # print(time.time() - t0)
    
    
    #some quick evaluation
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