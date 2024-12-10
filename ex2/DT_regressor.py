#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import random

#vgl. 03, 04, 06, 07

class DTRegressor():
    
    """
    Parameters:
        max_depth: max depth of decision tree. 0 for Zero Rule, 1 for One Rule, -1 for unlimited
        compute_split_alg: the algorithm used to compute the best split of the data
            should be one out of "error_rate", "information_gain", "gini_index", "variance_reduction", ???
            TODO: actually, might neeed to different, e.g. means squared error (MSE), mean absolute error (MAE)
        epsilon: stop early if standard deviation of prediction is smaller than this
        TODO: something about (pre-)pruning
        random_state: parameter for reproducability for eventual random operations; default=None
        splitter: 'best' to choose the best from all possible splits at each step, 
            'random' to choose the best split from max_features random features; default: 'best'
        max_features: upper limit to how many features are selected when computing the split. 
            Only has an effect if splitter='random' and max_features < sample features; default: None
    """
    def __init__(self, max_depth=-1, 
            compute_split_alg="mse",
            epsilon=0.001,
            random_state=None,
            splitter='best',
            max_features=None):
        self.max_depth = max_depth
        self.tree_root = None
        self.epsilon = epsilon
        
        if random_state == None:
            self.rd = random.Random()
        else:
            self.rd = random.Random(random_state)
            
        #error measurement used
        if compute_split_alg == "mse":
            self.goodness_test = self._mse
        elif compute_split_alg == "mae":
            self.goodness_test = self._mae
        else:
            raise ValueError("invalid goodness commputation method selected")
    
        self.splitter = splitter
        self.max_features = max_features

    
    """
        trains the model
        X should be a numpy array of shape(number_of_samples, number_of_features)
    """
    def fit(self, X, y):
        if self.max_depth == 0:
            #Zero Rule
            self.tree_root = self.LeafNode(np.mean(y,axis=0))
        else:
            self.tree_root = self._fit(X, y, 0)

        print("*"*10+"Training finished"+"*"*10)
                    
    """
        X should be a numpy array of shape(number_of_samples, number_of_features)
        should be able to correctly learn multiple regression values at the same time 
            (more than one target column)
    """
    def _fit(self, X, y, depth):
        # if X.ndim == 1: 
        if X.shape[0] == 1:
            #only one sample (row) left, we want to predict its y-value(s) 
            return self.LeafNode(y)
        elif depth == self.max_depth:
            #max depth reached, we want to predict the mean(s) of its y-values 
            # print("max depth reached")
            return self.LeafNode(np.mean(y,axis=0))
        elif np.max(np.sqrt(self._mse(y.mean(axis=0), y))) < self.epsilon:  
            #standard deviation for (all) predicted values is smaller than our target value, stop early
            #so for all X_is the average of the corresponding y_is is not too far from the individual y_is
            return self.LeafNode(np.mean(y,axis=0))
        else:
            # return an inner node with children set by recursive calls to _fit(), with i-1 and split X, y 
            #    --> divide X based on whatever splitting criterion I calculate, then pass only the respective parts of X and the corresponding parts of y to the repective recursive calls
            test = self._create_split(X, y)
            # masks = [[test(X_i) == j for X_i in X] for j in range(self.split_on_decision)] #split_on_decision is no longer used, delete this
            masks = [[test(X_i) == j for X_i in X] for j in range(2)] #TODO: could probably make this more efficient by better utilizing numpy
            # print(X)
            # print(X.ndim)
            # print(X.shape)
            # print(masks)
            # print(i)
            children = [self._fit(X[mask,:], y[mask], depth+1) for mask in masks] #split X, y by selecting just certain parts
            return self.InnerNode(test, children)
        
    def _create_split(self, X, y):
        # best_split = lambda X_i: 0
        best_error = float("inf")
        best_column_to_split = 0
        best_feature_value_to_split_on = 0
        
        # print("------ calculating new split ------")
        
        #initial brute force attempt = 'best' strategy
        if self.splitter == 'best': #TODO might randomly shuffle the features here too before computing split
            for row, sample in enumerate(X):
                for column, feature_value in enumerate(sample):
                    # print("sample, feature:")
                    # print(sample)
                    # print(feature)
                    # print("comparing colum %d" % j)
                    #split on feature
                    mask_0 = X[:, column] < feature_value
                    X_0 = X[mask_0]
                    y_0 = y[mask_0]
                    
                    # mask_1 = X[:, column] >= feature_value
                    mask_1 = np.invert(mask_0)
                    X_1 = X[mask_1]
                    y_1 = y[mask_1]
                    
                    # print("X_0, X_1")
                    # print(X_0)
                    # print(X_1)
                    if X_0.shape[0] == 0 or X_1.shape[0] == 0:
                        #we have hit an outermost value, a split is empty so not good
                        # print("split no good")
                        continue
                        
    
                    cur_y_predict_0 = np.mean(y_0,axis=0)
                    cur_y_predict_1 = np.mean(y_1,axis=0)
                    
                    error_0 = np.max(self.goodness_test(cur_y_predict_0, y_0)) #max in case there are multiple predicted values
                    error_1 = np.max(self.goodness_test(cur_y_predict_1, y_1))
                    error = max(error_0, error_1)
                    # print(error)
                    if error < best_error: #better error
                        # best_split = lambda X_i: 0 if X_i[j] < feature else 1
                        best_column_to_split = column
                        best_feature_value_to_split_on = feature_value
                        best_error = error
        elif self.splitter == "random":
            raise ValueError("not implemented yet") #TODO
            
        else:
            raise ValueError("invalid splitter set")

        # print("best error: %f" % best_error)
        # print("best column to split: %d" % best_column_to_split)
        # print("best feature value to split on: %f" % best_feature_value_to_split_on)
        # print(best_error)
        return lambda X_i: 0 if X_i[best_column_to_split] < best_feature_value_to_split_on else 1
        # return best_split

        
    
    """
        predicts and returns y for the given X
    """
    def predict(self, X):
        
        if self.tree_root == None:
            raise AttributeError("Tree has not been trained. No root set")
        else:
            y = []
            stack_method = None #for selecting how the single predictions should be put together
            # axis = -1
            
            for X_i in X:
                prediction = self.tree_root.get_prediction(X_i)
                
                if stack_method == None: #set stack_method once
                    if prediction.shape == (1,): #return value should be of shape (m,)
                        stack_method = np.hstack
                    else: #return value should be of shape (m, n)
                        stack_method = np.vstack
                # if axis < 0:
                    # if prediction.shape == (1,): #return value should be of shape (m,)
                        # axis = 1
                    # else: #return value should be of shape (m, n)
                        # axis = 0
                        
                y.append(prediction)
                # print(prediction)
            return stack_method(y)
            # return np.stack(y, axis=axis)
    
    """
    calculates the mean squared error for two vectors (or two matrices for each of the respective columns)
    """
    def _mse(self, prediction_matrix, y):
        return ((prediction_matrix - y)**2).mean(axis=0)
        
    """
    calculates the mean absolute error for two vectors (or two matrices for each of the respective columns)
    """
    def _mae(self, prediction_matrix, y):
        return (np.abs(prediction_matrix - y)).mean(axis=0)
                
    
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
            #use test to get the prediction value of the relevant of the child node
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
    # y = np.array([0.5,0.6])
    # y = np.array([[0.5],[0.6]])
    y = np.array([[0.5,0.6], 
                  [0.6,0.7]])
    # reg = DT_Regressor(max_depth=0)
    # reg = DT_Regressor(max_depth=1)
    reg = DTRegressor()
    reg.fit(X, y)
    print(reg.predict(X))
    print(reg.predict(X).shape)

if __name__ == '__main__':
    main()