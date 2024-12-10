#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#vgl 06
import random
import numpy as np

from DT_regressor import DTRegressor
from sklearn.tree import DecisionTreeRegressor

class RandomForestRegressor():
    
    
    """
        use_skl_tree: should the random forest be built from trees from sklearn? For testing/comparsion purposes, default: False
    """
    def __init__(self, use_skl_tree=False, num_trees=100, max_depth=-1, random_state=None, criterion='mse'):
        if not use_skl_tree:
            self.TreeClass = DTRegressor
        else:
            self.TreeClass = DecisionTreeRegressor
            
        if criterion == 'squared_error':
            self.criterion = 'mse'
        elif criterion == 'absolute_error':
            self.criterion = 'mae'
        else:
            raise ValueError("criterion not implemented")
        
        self.random_state = random_state
        if random_state == None:
            self.rd = random.Random()
        else:
            self.rd = random.Random(random_state)
            
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.trees = []
            
    def fit(self, X, y):
        # 1. create multiple data sets
        data_sets = []
        for i in range(self.num_trees):
            # select a number of samples (bootstrapping, selection with replacement) for each data set (TODO: how many - fixed/random?)
            # we want to select indices from range(num_samples), then add tuples (X[these_indices], y[these_indices])
            pass
        # 2. buld multiple classifiers
        self.trees = []
        for i, data_set in enumerate(data_sets):
            clf = self.TreeClass(
                max_depth=self.max_depth,
                random_state=None if self.random_state == None else self.random_state+i,
                # splitter='random', #TODO: uncomment this
                compute_split_alg=self.citerion
                )
            clf.fit(data_set[0], data_set[1])
            self.trees.append(clf)
        # 3. combine classifiers
        # done only when predicting
        
    def predict(self, X):
        if len(self.trees) == 0:
            raise AttributeError("Forest has not been trained. No trees available")
        else:
            predictions = np.array([tree.predict(X) for tree in self.trees])
            # combine predictions 
            prediction = np.mean(predictions, axis=0) #simple mean over the predicted results
            # prediction = np.median(predictions, axis=0) #simple median over the predicted results, might allow seleting this
            return prediction
        
        
            
        

def main():
    pass



if __name__ == '__main__':
    main()