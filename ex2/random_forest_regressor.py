#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#vgl 06
import random
import numpy as np
# import pandas as pd

from DT_regressor import DTRegressor
from sklearn.tree import DecisionTreeRegressor

class RandomForestRegressor():
    
    """
        use_skl_tree: should the random forest be built from trees from sklearn? For testing/comparsion purposes, default: False
        n_estimators:
        max_depth:
        random_state:
        criterion:
        max_samples:
        max_features:
        bootstrap:
        min_samples_split:
        min_samples_leaf:
        max_leaf_nodes:
        
    """
    def __init__(self, use_skl_tree=False,
                 n_estimators=100, 
                 max_depth=None, 
                 random_state=None, 
                 criterion='squared_error',
                 max_samples=None,
                 max_features=1.0,
                 bootstrap=True,
                 min_samples_split=2, 
                 min_samples_leaf=1,
                 max_leaf_nodes=None,
                 vote='mean'):
        if not use_skl_tree:
            self.TreeClass = DTRegressor
        else:
            self.TreeClass = DecisionTreeRegressor
            
        # if criterion == 'squared_error':
        #     self.criterion = 'mse'
        # elif criterion == 'absolute_error':
        #     self.criterion = 'mae'
        # else:
        #     raise ValueError("criterion not implemented")
        self.criterion = criterion
        
        self.random_state = random_state
        if random_state == None:
            self.rd = random.Random()
        else:
            self.rd = random.Random(random_state)
            
        self.num_trees = n_estimators
        self.max_depth = max_depth
        self.trees = []
        
        self.max_samples_in_tree = max_samples
        self.max_features = max_features
        
        self.bootstrap = bootstrap #TODO: make this do something
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_leaf_nodes = max_leaf_nodes 
        
        self.vote = vote
            
    def fit(self, X, y):
        try: #we don't need pandas dataframes, we want np arrays
            X = X.to_numpy()
            y = y.to_numpy()
        except AttributeError:
            pass
        
        
        # 1. create multiple data sets
        data_sets = []
        indices = [i for i in range(X.shape[0])]
        for i in range(self.num_trees):
            # select a number of samples (bootstrapping, selection with replacement) for each data set (TODO: how many - fixed/random?)
            # we want to select indices from range(num_samples), then add tuples (X[these_indices], y[these_indices])    
            self.rd.shuffle(indices)
            selected_indices = self._select_indices(self.max_samples_in_tree, indices)

            data_sets.append((X[selected_indices], 
                              y[selected_indices]))
        # 2. buld multiple classifiers
        self.trees = []
        for i, data_set in enumerate(data_sets):
            clf = self.TreeClass(
                max_depth=self.max_depth,
                random_state=None if self.random_state == None else self.random_state+i,
                splitter='random', 
                # compute_split_alg=self.criterion,
                criterion=self.criterion,
                max_features=self.max_features,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                max_leaf_nodes=self.max_leaf_nodes
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
            if self.vote == 'mean':
                prediction = np.mean(predictions, axis=0) #simple mean over the predicted results
            elif self.vote == 'median':
                prediction = np.median(predictions, axis=0) #simple median over the predicted results, might allow seleting this
            else:
                raise ValueError("no valid voting method specified")
            return prediction
        
    """
    a little helper method to only return the first n indices
    where n is determined in some way by num_to_select
    """
    def _select_indices(self, num_to_select, indices):

        if num_to_select == None:
            selected_indices = indices
        elif isinstance(num_to_select, float):
            max_index = int(num_to_select * len(indices))
            selected_indices = indices[:max_index]
        elif num_to_select == 'sqrt':
            max_index = int(np.sqrt(len(indices)))
            selected_indices = indices[:max_index]
        elif num_to_select == 'log2':
            max_index = int(np.log2(len(indices)))
            selected_indices = indices[:max_index]
        else:
            selected_indices = indices[:num_to_select]
        return selected_indices

        
            
        

def main():
    #for some basic sanity testing
    X = np.array([[1,2, 5], [3, 4,8]])
    # y = np.array([0.5,0.6])
    # y = np.array([[0.5],[0.6]])
    y = np.array([[0.5,0.6], 
                  [0.6,0.7]])
    reg = RandomForestRegressor()
    reg.fit(X, y)
    print(reg.predict(X))
    print(reg.predict(X).shape)



if __name__ == '__main__':
    main()