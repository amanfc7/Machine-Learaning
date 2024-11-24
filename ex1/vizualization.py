#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pprint import pprint
from ds_load_util import load_dataset

import wine_class_mlp
import second_ds_class_mlp
import congress_class_mlp
import reviews_class_mlp

import sys


def main():
    
    best_wine_class_mlp_clf = wine_class_mlp.train_model()
    best_second_class_mlp_clf = second_ds_class_mlp.train_model()
    best_congress_class_mlp_clf = congress_class_mlp.train_model()
    best_reviews_class_mlp_clf = reviews_class_mlp.train_model()

    best_wine_class_dt_clf = None
    best_second_class_dt_clf = None
    best_congress_class_dt_clf = None
    best_reviews_class_dt_clf = None

    best_wine_class_gpc_clf = None
    best_second_class_gpc_clf = None
    best_congress_class_gpc_clf = None
    best_reviews_class_gpc_clf = None




if __name__ == '__main__':
    main()