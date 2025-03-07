#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
# import pandas as pd
import random
from queue import PriorityQueue

#vgl. 03, 04, 06, 07

class DTRegressor():
    
    """
    Parameters:
        max_depth: max depth of decision tree. 0 for Zero Rule, 1 for One Rule, -1 for unlimited
        criterion: the algorithm used to compute the best split of the data
            should be one out of means squared error ('squared_error'), mean absolute error ('absolute_error')
        epsilon: stop early if standard deviation of prediction is smaller than this
        random_state: parameter for reproducability for eventual random operations; default=None
        splitter: 'best' to choose the best from all possible splits at each step, 
            'random' to choose the best split from max_features random features; default: 'best'
        max_features: upper limit to how many features are selected when computing the split. 
            Only has an effect if splitter='random' and max_features < sample features; default: None
        min_samples_split: if set gives a a lower limit to the amount of samples that should be part of each leaf; default: 2
        min_samples_leaf: if set gives a a lower limit to the amount of samples that should be part of each leaf; default: 1
        max_leaf_nodes: if set to an integer, only up to this amount of leaves will be created in total; default: None
        verbose: if True, print some details about the tree once training has finished; default: False
        
    """
    def __init__(self, 
                 max_depth=-1,
                 criterion="squared_error",
                 epsilon=0.001,
                 random_state=None,
                 splitter='best',
                 max_features=None,
                 min_samples_split=2, 
                 min_samples_leaf=1,
                 max_leaf_nodes=None,
                 verbose=False):
        self.max_depth = -1 if max_depth == None else max_depth
        self.tree_root = None
        self.epsilon = epsilon
        
        if random_state == None:
            self.rd = random.Random()
        else:
            self.rd = random.Random(random_state)
            
        #error measurement used
        if criterion == "squared_error":
            self.goodness_test = self._variance_reduction
        elif criterion == "absolute_error":
            self.goodness_test = self._absolute_reduction
        else:
            raise ValueError("invalid goodness commputation method selected")
    
        self.splitter = splitter
        self.max_features = max_features
        
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_leaf_nodes = max_leaf_nodes

        self.verbose = verbose

    
    """
        trains the model
        X should be a numpy array of shape(number_of_samples, number_of_features)
    """
    def fit(self, X, y):
        try: #we don't need pandas dataframes, we want np arrays
            X = X.to_numpy()
            y = y.to_numpy()
        except AttributeError:
            pass

        self.computed_min_samples_leaf = self._compute_min_samples_number(self.min_samples_leaf, X.shape[0])
        self.computed_min_samples_split = self._compute_min_samples_number(self.min_samples_split, X.shape[0])
        
        if self.max_depth == 0:
            #Zero Rule
            self.tree_root = self.LeafNode(np.mean(y,axis=0))
            self.depth = 0
            self.num_leaves = 1
        else:
            if self.max_leaf_nodes == None:
                self.tree_root, self.depth_reached, self.num_leaves = self._fit_rec(X, y, 0)
            else:
                self.tree_root, self.depth_reached, self.num_leaves = self._fit_qu(X, y)
            

        if self.verbose:
            print("*"*10+" Tree training finished "+"*"*10)
            print("depth reached: %d" % self.depth_reached)
            print("leaves created: %d" % self.num_leaves)
                    
    """
        X should be a numpy array of shape(number_of_samples, number_of_features)
        should be able to correctly learn multiple regression values at the same time 
            (more than one target column)
    """
    def _fit_rec(self, X, y, depth):
        ret_depth = depth
        ret_num_leaves = 1
        if X.shape[0] == 1:
            #only one sample (row) left, we want to predict its y-value(s) 
            node = self.LeafNode(y)
        elif X.shape[0] < self.computed_min_samples_split:
            #there are less samples left than we want to split further
            node = self.LeafNode(np.mean(y,axis=0))
        elif depth == self.max_depth:
            #max depth reached, we want to predict the mean(s) of its y-values 
            # print("max depth reached")
            node = self.LeafNode(np.mean(y,axis=0))
        elif np.max(np.sqrt(self._mse(y.mean(axis=0), y))) < self.epsilon:  
            #standard deviation for (all) predicted values is smaller than our target value, stop early
            #so for all X_is the average of the corresponding y_is is not too far from the individual y_is
            node = self.LeafNode(np.mean(y,axis=0))
            # print("early stop from small SD")
        else:
            # return an inner node with children set by recursive calls to _fit(), with i-1 and split X, y 
            #    --> divide X based on whatever splitting criterion I calculate, then pass only the respective parts of X and the corresponding parts of y to the repective recursive calls
            try:
                X_0, y_0, X_1, y_1, test = self._create_split(X, y)
                
                child_0, depth_0, num_nodes_0 = self._fit_rec(X_0, y_0, depth+1)
                child_1, depth_1, num_nodes_1 = self._fit_rec(X_1, y_1, depth+1)
                node = self.InnerNode(test, [child_0, child_1])
                ret_depth = max(depth_0, depth_1)
                ret_num_leaves = num_nodes_0 + num_nodes_1
            except TypeError:
                # all remaining samples have the same values in all features, we can't make a reasonable split, just take the average
                node = self.LeafNode(np.mean(y,axis=0))
            
        
        return (node, ret_depth, ret_num_leaves)
    
    def _fit_qu(self, X, y):
        #put inner nodes to expand next in a queue, based on how much splitting them reduces variance?
        depth_reached = 0
        min_leaves_created = 1

        if X.shape[0] == 1:
            #only one sample (row) left, we want to predict its y-value(s) 
            node = self.LeafNode(y)
            return (node, depth_reached, min_leaves_created)
        elif X.shape[0] < self.computed_min_samples_split:
            #there are less samples left than we want to split further
            node = self.LeafNode(np.mean(y,axis=0))
            return (node, depth_reached, min_leaves_created)
        elif np.max(np.sqrt(self._mse(y.mean(axis=0), y))) < self.epsilon:  
            #standard deviation for (all) predicted values is smaller than our target value, stop early
            #so for all X_is the average of the corresponding y_is is not too far from the individual y_is
            node = self.LeafNode(np.mean(y,axis=0))
            # print("early stop from small SD")
            return (node, depth_reached, min_leaves_created)
        elif self.max_leaf_nodes <= min_leaves_created:
            node = self.LeafNode(np.mean(y,axis=0))
            # print("early stop from creating too many leaves")
            return (node, depth_reached, min_leaves_created)

        root_node = self.InnerNode(None, [None, None])
        queue = PriorityQueue()
        try:
            queue.put(self._create_queue_item(X, y, root_node, depth_reached+1))
        except TypeError:
            # all remaining samples have the same values in all features, we can't make a reasonable split, just take the average
            node = self.LeafNode(np.mean(y,axis=0))
            return (node, depth_reached, min_leaves_created)

        while not queue.empty() and min_leaves_created < self.max_leaf_nodes:
            _, (X_0, y_0, X_1, y_1, test, _, parent_node, at_depth) = queue.get() #this picks the split resulting in the larges variance reduction
            # we want to fill the parent node and then add more splits to the queue
            parent_node.set_test(test)
            if parent_node.test == None:
                print("ALERT: test = none")
            depth_reached = np.max([depth_reached, at_depth])

            if X_0.shape[0] == 1:
                #only one sample (row) left, we want to predict its y-value(s) 
                parent_node.set_child(0, self.LeafNode(np.mean(y_0,axis=0)))
            elif X_0.shape[0] < self.computed_min_samples_split:
            #there are less samples left than we want to split further
                parent_node.set_child(0, self.LeafNode(np.mean(y_0,axis=0)))
            elif np.max(np.sqrt(self._mse(y_0.mean(axis=0), y_0))) < self.epsilon:  
                #standard deviation for (all) predicted values is smaller than our target value, stop early
                parent_node.set_child(0, self.LeafNode(np.mean(y_0,axis=0)))
            else:
                try:
                    child_0 = self.InnerNode(None, [None, None], parent_node=parent_node, parent_child_index=0)
                    parent_node.set_child(0, child_0)
                    queue.put(self._create_queue_item(X_0, y_0, child_0, depth_reached+1))
                except TypeError:
                    # all remaining samples have the same values in all features, we can't make a reasonable split, just take the average
                    parent_node.set_child(0, self.LeafNode(np.mean(y_0,axis=0)))

            if X_1.shape[0] == 1:
                #only one sample (row) left, we want to predict its y-value(s) 
                parent_node.set_child(1, self.LeafNode(np.mean(y_1,axis=0)))
            elif X_1.shape[0] < self.computed_min_samples_split:
            #there are less samples left than we want to split further
                parent_node.set_child(1, self.LeafNode(np.mean(y_1,axis=0)))
            elif np.max(np.sqrt(self._mse(y_1.mean(axis=0), y_1))) < self.epsilon:  
                #standard deviation for (all) predicted values is smaller than our target value, stop early
                parent_node.set_child(1, self.LeafNode(np.mean(y_1,axis=0)))
            else:
                try:
                    child_1 = self.InnerNode(None, [None, None], parent_node=parent_node, parent_child_index=1)
                    parent_node.set_child(1, child_1)
                    queue.put(self._create_queue_item(X_1, y_1, child_1, depth_reached+1))
                except TypeError:
                    # all remaining samples have the same values in all features, we can't make a reasonable split, just take the average
                    parent_node.set_child(1, self.LeafNode(np.mean(y_1,axis=0)))

            min_leaves_created = min_leaves_created + 1

        if not queue.empty():
            #there are still some items in the queue, we need to turn empty inner nodes into leaves
            while not queue.empty():
                _, (_, _, _, _, _, y_mean, parent_node, at_depth) = queue.get() #this picks the split resulting in the larges variance reduction
                parent_node.parent.set_child(parent_node.parent_child_index, self.LeafNode(y_mean))

        return (root_node, depth_reached, min_leaves_created)
        

    def _create_queue_item(self, X, y, parent_node, at_depth):
        X_0, y_0, X_1, y_1, test = self._create_split(X, y)
        variance_redcution_from_candidate = self.goodness_test(y, y_0, y_1)
        return (-variance_redcution_from_candidate, (X_0, y_0, X_1, y_1, test, np.mean(y,axis=0), parent_node, at_depth))
        
        
    def _create_split(self, X, y):
        
        # print("------ calculating new best split ------")
        
        #create a random permutation for the feature indices
        indices = [i for i in range(X.shape[1])]
        self.rd.shuffle(indices)

        
        #initial brute force attempt = 'best' strategy
        if self.splitter == 'best':
            best_column_to_split, best_feature_value_to_split_on  = self._create_candidate_split(X, y, indices)

        
        #we might want to look at only the first n features based on max_features instead
        elif self.splitter == "random":
            indices = self._select_indices(self.max_features, indices)
            best_column_to_split, best_feature_value_to_split_on  = self._create_candidate_split(X, y, indices)
            
        else:
            raise ValueError("invalid splitter set")

    
        mask_0 = X[:, best_column_to_split] < best_feature_value_to_split_on
        X_0 = X[mask_0]
        y_0 = y[mask_0]
        
        mask_1 = np.invert(mask_0)
        X_1 = X[mask_1]
        y_1 = y[mask_1]
        test = lambda X_i: 0 if X_i[best_column_to_split] < best_feature_value_to_split_on else 1
        return (X_0, y_0, X_1, y_1, test)


    def _create_candidate_split(self, X, y, indices):
        best_var_red = -float("inf")
        best_column_to_split = None
        best_feature_value_to_split_on = None

        
        for column in indices:
            features_of_column = X[:, column]
            features_of_column = np.unique(X[:, column]) #
            # features_of_column.sort() #no improvement: n log n vs 2n
            # for feature_value in features_of_column[1:]:
            for feature_value in features_of_column:

                #split on feature
                mask_0 = X[:, column] < feature_value
                y_0 = y[mask_0]
                
                if y_0.shape[0] == 0 or y_0.shape[0] == y.shape[0]:
                    #we have hit an outermost value in the range for this feature, one split is empty so not good
                    # print("split no good")
                    continue

                if y_0.shape[0] < self.computed_min_samples_leaf or (y.shape[0] - y_0.shape[0]) < self.computed_min_samples_leaf:
                 #the split does not leave enough samples in a leaf
                    # print("split no good")
                    continue
                
                mask_1 = np.invert(mask_0)
                y_1 = y[mask_1]

                    
                variance_redcution_from_candidate = self.goodness_test(y, y_0, y_1)
                
                if variance_redcution_from_candidate > best_var_red:
                    best_var_red = variance_redcution_from_candidate
                    best_column_to_split = column
                    best_feature_value_to_split_on = feature_value

                    
         
        return (best_column_to_split, best_feature_value_to_split_on)


    
    """
        predicts and returns y for the given X
    """
    def predict(self, X):
        try: #we don't need pandas dataframes, we want np arrays
            X = X.to_numpy()
        except AttributeError:
            pass
        
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
    
    """
        computes the variance reduction for a y and a given split into y_0 and y_1
    """
    def _variance_reduction(self, y, y_0, y_1):
        samples_num_0 = len(y_0)
        samples_num_1 = len(y_1)
        samples_num_tot = len(y)
        y_variance = self._mse(np.mean(y,axis=0), y)
        y_0_variance = self._mse(np.mean(y_0,axis=0), y_0)
        y_1_variance = self._mse(np.mean(y_1,axis=0), y_1)
        
        variance_reduction = y_variance - (samples_num_0 / samples_num_tot * y_0_variance + samples_num_1 / samples_num_tot * y_1_variance)
        return np.max(variance_reduction) #max in case of multiple values to predict
    
    def _absolute_reduction(self, y, y_0, y_1):
        samples_num_0 = len(y_0)
        samples_num_1 = len(y_1)
        samples_num_tot = len(y)
        y_variance = self._mae(np.mean(y,axis=0), y)
        y_0_variance = self._mae(np.mean(y_0,axis=0), y_0)
        y_1_variance = self._mae(np.mean(y_1,axis=0), y_1)
        
        abs_red =  y_variance - (samples_num_0 / samples_num_tot * y_0_variance + samples_num_1 / samples_num_tot * y_1_variance)
        return np.max(abs_red) #max in case of multiple values to predict
    
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
    
        
    """
        a little helper method to  convert the min_samples int or float into a proper int
    """
    def _compute_min_samples_number(self, min_samples, num_samples):
        if isinstance(min_samples, float):
            return np.ceil(min_samples * num_samples)
        elif isinstance(min_samples, int):
            return min_samples
        else:
            raise ValueError("Invalid value for min_samples_*")
        
                
    
    class TreeNode():
        
        def __init__(self, children):
            self.children = children
        
        def get_prediction(self, X_i):
            pass
        
        
    class InnerNode(TreeNode):
        
        """
        test should be a function that takes a vector/list and based on that 
            returns a non-negative integer < len(children)
            test should be set during the training step
        """

        def __init__(self, test, children, parent_node=None, parent_child_index=None):
            super().__init__(children=children)
            self.test = test
            self.parent = parent_node
            self.parent_child_index = parent_child_index

        # def __init__(self, test, children):
        #     self.__init__(test, children, None, None)

        
        
        def get_prediction(self, X_i):
            #use test to get the prediction value of the relevant of the child node
            return self.children[self.test(X_i)].get_prediction(X_i)
        
        def set_child(self, i, child):
            self.children[i] = child

        def set_test(self, test):
            self.test = test
        
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
    #for some basic sanity testing
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