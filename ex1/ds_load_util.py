#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ucimlrepo import fetch_ucirepo
from sklearn.datasets import fetch_openml, load_wine
# from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


import numpy as np
import pandas as pd
import sys

def main():
    load_dataset(sys.argv[1], verbose=True)


def load_dataset(name,preprocess=False,
                 encoder=None,
                 imputer=None,
                 scaler=None,
                 verbose=False
                 ):
    
    if name == 'wine':
        wine = load_wine()
        X = wine.data
        y = wine.target
    elif name == 'mushroom':
        ds = fetch_openml(name='mushroom', version=1)
        X = ds.data
        X = X.where(X!='?', other=np.nan)
        y = ds.target
    elif name == 'congress':
        ds = pd.read_csv('CongressionalVotingID.shuf.lrn.csv')
        if verbose:
            print(ds.sample(3))
        X = ds.drop(['ID', 'class'], axis=1)
        X = X.where(X!='unknown', other=np.nan)
        y = ds['class']
    elif name == 'reviews':
        ds = pd.read_csv('amazon_review_ID.shuf.lrn.csv')
        # print(ds)
        # ds_test = pd.read_csv('amazon_review_ID.shuf.tes.csv')
        # print(ds)
        if verbose:
            print(ds.sample(3))
        X = ds.drop(['ID', 'Class'], axis=1)
        y = ds['Class']
        
        # print(X)
        # print(y)
        # print(X.max(axis=0).min()) #=1, so there are no entirely empty colums
    elif name =='adult':
        ds = fetch_ucirepo(id=2)
        X = ds.data.features
        X = X.where(X!='NaN', other=np.nan)
        y = np.ravel(ds.data.targets)
        if verbose:
            print(ds.metadata)
            print(ds.variables)
            # print(y)
            # print(np.ravel(y).shape)
    elif name == 'second':
        ds = fetch_ucirepo(id=365)
        X = ds.data.features
        X = X.where(X!='NaN', other=np.nan)
        y = ds.data.targets
        y = np.ravel(y)
        if verbose:
            print(ds.metadata)
            print(ds.variables)
            # print(y)
            # print(np.ravel(y).shape)
    elif name == 'second2':
        ds = fetch_openml(name='sick', version=1)
        X = ds.data.drop(['TBG', 'TBG_measured'], axis=1) #can be dropped since the latter is monovalued ('f') and the former only contais missing values
        # X = X.where(X!='?', other=np.nan)
        y = ds.target
        # X = ds.data.features
        # X = X.where(X!='NaN', other=np.nan)
        # y = ds.data.targets
        # y = np.ravel(y)
        if verbose:
            # print(ds)
            print(ds.data.columns)
            # print(ds.metadata)
            # print(ds.variables)
            # print(y)
            # print(np.ravel(y).shape)
    elif name == 'second3':
        ds = fetch_ucirepo(id=536)
        X = ds.data.features
        X = X.where(X!='NaN', other=np.nan)
        y = ds.data.targets
        # y = np.ravel(y)
        if verbose:
            print(ds)
            print(ds.data)
            # print(ds.metadata)
            # print(ds.variables)
            print(y)
            # print(np.ravel(y).shape)
    else:
        print("unknown Dataset")
        return (0,0,0,0)
    
    if not preprocess:
        return train_test_split(
            X, y, test_size=0.3, random_state=0)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)

    pipeline_steps = []
    if encoder != None:
        pipeline_steps.append(("encoder", encoder))
    if imputer != None:
        pipeline_steps.append(('imputer', imputer))
    if scaler != None:
        pipeline_steps.append(('scaler', scaler))
        
    if len(pipeline_steps) > 0:
        enc = Pipeline(steps=pipeline_steps)
        X_train = enc.fit_transform(X_train)
        X_test = enc.transform(X_test)
    
    return (X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()