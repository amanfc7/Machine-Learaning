#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

#vgl. 03, 04, 06, 07

class DT_Regressor():
    
    """
    Parameters:
        max_depth: max depth of decision tree. 0 for Zero Rule, 1 for One Rule, -1 for unlimited
        split_on_decision: how many branches per inner node, min 2
        compute_split_alg: the algorithm used to compute the best split of the data
            should be one out of "error_rate", "information_gain", "gini_index", "variance_reduction", ???
            TODO: actually, might neeed to different, e.g. means squared error (MSE), mean absolute error (MAE)
        TODO: something about (pre-)pruning
    """
    def __init__(self, max_depth=-1, split_on_decision=2, compute_split_alg="error_rate"):
        self.max_depth = max_depth
        self.split_on_decision = split_on_decision
        self.tree_root = None
    
    

    
    """
        trains the model
        X should be a numpy array of shape(number_of_samples, number_of_features)
    """
    def fit(self, X, y):
        if self.max_depth == 0:
            #Zero Rule
            self.tree_root = self.LeafNode(np.mean(y,axis=0))
        else:
            self.tree_root = self._fit(X, y, self.max_depth)
            # i = self.max_depth
            # while i != 0:
                # i -= 1
                ## add some inner nodes
                # #probably want to do this recursively, actually, would not need below then
            
                # if i == 0 or False: #TODO: False should be replaced with only one value remaining in box
                    ## add leaf nodes
                    # pass
                    
    """
        X should be a numpy array of shape(number_of_samples, number_of_features)
        should be able to correctly learn multiple regression values at the same time 
            (more than one target column)
    """
    def _fit(self, X, y, i):
        if X.ndim == 1: 
            #only one sample left, we want to predict its y-value(s) 
            return self.LeafNode(y)
        elif i == 0:
            #max depth reached, we want to predict the mean(s) of its y-values 
            return self.LeafNode(np.mean(y,axis=0))
        else:
            # return an inner node with children set by recursive calls to _fit(), with i-1 and split X, y 
            #    --> divide X based on whatever splitting criterion I calculate, then pass only the respective parts of X and the corresponding parts of y to the repective recursive calls
            test = self._create_split(X, y)
            children = [self._fit(X, y, i-1) for j in range(self.split_on_decision)] #TODO: split X, y
            return self.InnerNode(test, children)
        
    def _create_split(self, X, y):
        return lambda X_i: 0
        # pass
        
    
    """
        predicts and returns y for the given X
    """
    def predict(self, X):
        y = [] 
        if self.tree_root == None:
            raise AttributeError("Tree has not been trained. No root set")
        else:
            for X_i in X:
                y.append(self.tree_root.get_prediction(X_i))
            return np.array(y)
                
    
    class TreeNode():
        
        def __init__(self, children):
            self.children = children
        
        def get_prediction(self, X_i):
            pass
        
        # def add_child_node(self, node):
        #     #prbly add node at index i in self.children, then increase local i by one, raise error if i >= num_children instead
        #     # could also try assignment directly and catch TypeError (Child node) or IndexError (Full inner node)
        #     #might instead want to add all children at once
        #     pass
        
    class InnerNode(TreeNode):
        
        """
        test should be a function that takes a vector/list and based on that 
            returns a non-negative integer < len(children)
            test should be set during the training step
        """
        def __init__(self, test, children):
            super().__init__(children=children)
            self.test = test
        
        def get_prediction(self, X_i):
            #use test to get the prediction value of one of the child nodes
            return self.children[self.test(X_i)].get_prediction(X_i)
        
    class LeafNode(TreeNode):
        
        """
            prediction value should be the value of the average 
                of all values contained in this leaf node or the only value
        """
        def __init__(self, prediction_value):
            super().__init__(children=None)
            self.prediction_value = prediction_value
        
        def get_prediction(self, X_i):
            #return own prediction value
            return self.prediction_value
    


def main():
    X = np.array([[1,2, 5], [3, 4,8]])
    y = np.array([0.5,0.6])
    # y = np.array([[0.5,0.6], 
    #               [0.6,0.7]])
    reg = DT_Regressor(max_depth=1)
    reg.fit(X, y)
    print(reg.predict(X))
    print(reg.predict(X).shape)

if __name__ == '__main__':
    main()