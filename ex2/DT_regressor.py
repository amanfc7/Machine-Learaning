#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
        pass
    
    
    """
        trains the model
    """
    def fit(self, X, y):
        pass
    
    
    def predict(self):
        pass
    
    
    class TreeNode():
        
        def __init__(self, num_children=2):
            pass
        
        def get_prediction(self, X_i):
            pass
        
    class InnerNode(TreeNode):
        
        """
        test should be a function that takes a vector/list and based on that returns a non-negative integer < num_children
        """
        def __init__(self, test, num_children=2):
            pass
        
        def get_prediction(self, X_i):
            #use test to get the prediction value of one of the child nodes
            # something like: return self.children[test(X_i)].get_prediction(X_i)
            pass
        
    class LeafNode(TreeNode):
        
        """
            prediction value should either get set to the value of the average of all values contained in this leaf node
        """
        def __init__(self, prediction_value):
            pass
        
        def get_prediction(self, X_i):
            #return ´own prediction value
            pass


def main():
    pass

if __name__ == '__main__':
    main()