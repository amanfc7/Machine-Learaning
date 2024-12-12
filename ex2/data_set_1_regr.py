#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from random_forest_regressor import RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import pandas as pd


def train_model():
    ds = pd.read_csv('colleges_preprocessed.csv')
    # if verbose:
    #     print(ds.sample(3))
    X = ds.drop(['percent_pell_grant'], axis=1)
    # X = X.where(X!='unknown', other=np.nan)
    y = ds['percent_pell_grant']
    
    # print(X)
    # print(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)
    
    clf = RandomForestRegressor(use_skl_tree=True)
    clf.fit(X_train, y_train)
    
    y_prediction = clf.predict(X_test)
    
    #some quick evaluation
    print(r2_score(y_test, y_prediction))
    

def main():
    train_model()
    



if __name__ == '__main__':
    main()