

from sklearn.datasets import fetch_openml, load_wine
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


import numpy as np
import pandas as pd

def load_dataset(name,preprocess=False,
                 encoder=None,
                 imputer=None,
                 scaler=None
                 ):
    
    if name == 'wine':
        wine = load_wine()
        X = wine.data
        y = wine.target
    elif name == 'congress':
        ds = pd.read_csv('CongressionalVotingID.shuf.lrn.csv')
        print(ds.sample(3))
        X = ds.drop(['ID', 'class'], axis=1)
        X = X.where(X=='unknown', other=np.nan)
        y = ds['class']
    elif name == 'reviews':
        ds = pd.read_csv('amazon_review_ID.shuf.lrn.csv')
        # print(ds)
        # ds_test = pd.read_csv('amazon_review_ID.shuf.tes.csv')
        # ds = fetch_openml(name='mushroom', version=1)
        # print(ds)
        print(ds.sample(3))
        X = ds.drop(['ID', 'Class'], axis=1)
        y = ds['Class']
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
        
    enc = Pipeline(steps=pipeline_steps)
    X_train = enc.fit_transform(X_train)
    X_test = enc.transform(X_test)
    
    return (X_train, X_test, y_train, y_test)
