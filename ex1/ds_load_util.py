#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ucimlrepo import fetch_ucirepo
from sklearn.datasets import fetch_openml, load_wine
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


import numpy as np
import pandas as pd
import sys

def main():
    load_dataset(sys.argv[1], verbose=True)


"""
    loads a data set by name and returns a train test split of data and target

    Parameters:
        * name: the name of the dataset to load
        * preprocess: boolean, should preprocessing be done, default=False
        * verbose: should some datails about the dataset get printed, default=False
        * encoder: an encoder object from sklearn, only used if preprocess=True, default=None
        * imputer: an imputer object from sklearn, only used if preprocess=True, default=None
        * scaler: a scaler object from sklearn. Only used if preprocess=True, default=None
        * scaler_no: a number, specifying a scaler to be used. If this is set, the scaler passed in scaler is not used. Only used if preprocess=True, default=None
"""
def load_dataset(name,
                 preprocess=False,
                 encoder=None,
                 imputer=None,
                 scaler=None,
                 scaler_no=None,
                 verbose=False
                 ):
    
    categories_to_transform = []

    if scaler_no != None:
        if scaler_no == 1:
            scaler = preprocessing.StandardScaler()
        elif scaler_no == 11:
            scaler = preprocessing.StandardScaler(with_mean=False)
        elif scaler_no == 2:
            scaler=preprocessing.MinMaxScaler()
        elif scaler_no == 3:
            scaler=preprocessing.RobustScaler()
        elif scaler_no == 4:
            scaler=preprocessing.MaxAbsScaler()
        else:
            print("Warning: Incalid scaler number. Using no scaler")
            scaler = None
    else:
        scaler = scaler
    
    if name == 'wine':
        wine = load_wine()
        X = wine.data
        y = wine.target
    elif name == 'mushroom': #unused data set
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
        if verbose:
            print(ds.sample(3))
        X = ds.drop(['ID', 'Class'], axis=1)
        y = ds['Class']
        
        # print(X)
        # print(y)
    elif name =='adult': #unused data set
        ds = fetch_ucirepo(id=2)
        X = ds.data.features
        X = X.where(X!='NaN', other=np.nan)
        y = np.ravel(ds.data.targets)
        if verbose:
            print(ds.metadata)
            print(ds.variables)
            # print(y)
            # print(np.ravel(y).shape)
    elif name == 'second': #unused data set
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
        y = ds.target
        X = X.where(X!='NaN', other=np.nan) # dunno if this is needed

        if verbose:
            print(ds.data)
            print(ds.data.columns)
            # print(ds.metadata)
            # print(ds.variables)
            # print(y)
            # print(np.ravel(y).shape)

        # in this data set, only some colums should get encoded
        categories_to_transform = ['sex', 'on_thyroxine', 'query_on_thyroxine',
                                    'on_antithyroid_medication','sick','pregnant',
                                    'thyroid_surgery','I131_treatment','query_hypothyroid',
                                    'query_hyperthyroid','lithium','goitre',
                                    'tumor','hypopituitary','psych','TSH_measured',
                                    'T3_measured','TT4_measured','T4U_measured',
                                    'FTI_measured','referral_source']
            
    elif name == 'second3': #unused data set
        ds = fetch_ucirepo(id=536)
        X = ds.data.features
        X = X.where(X!='NaN', other=np.nan)
        y = ds.data.targets
        # y = np.ravel(y)
        if verbose:
            print(ds)
            
            print(ds.data)
            print(X)
            # print(ds.metadata)
            # print(ds.variables)
            # print(y)
            # print(np.ravel(y).shape)
    else:
        print("unknown Dataset")
        return (0,0,0,0)
    
    if not preprocess:
        return train_test_split(
            X, y, test_size=0.3, random_state=0)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)
    
    if len(categories_to_transform) > 0:
        #we only want to encode some select categories
        if encoder != None:
            enc = encoder
            encoder = None
        else:
            enc = preprocessing.OrdinalEncoder()
        X_train[categories_to_transform] = enc.fit_transform(X_train[categories_to_transform])
        X_test[categories_to_transform] = enc.transform(X_test[categories_to_transform])

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