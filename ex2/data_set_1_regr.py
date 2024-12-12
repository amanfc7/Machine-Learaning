#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from random_forest_regressor import RandomForestRegressor

import pandas as pd


def train_model():
    ds = pd.read_csv('colleges_preprocessed.csv')
    # if verbose:
    #     print(ds.sample(3))
    X = ds.drop(['percent_pell_grant'], axis=1)
    # X = X.where(X!='unknown', other=np.nan)
    y = ds['percent_pell_grant']
    
    print(X)
    print(y)
    
    clf = RandomForestRegressor()
    clf.fit(X, y)
    
    #some evaluation

def main():
    train_model()
    



if __name__ == '__main__':
    main()