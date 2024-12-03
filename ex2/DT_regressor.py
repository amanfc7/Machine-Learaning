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
        self.max_depth = max_depth
        self.split_on_decision = split_on_decision
        self.tree_root = None
    
    
    """
        trains the model
    """
    def fit(self, X, y):
        pass
    
    """
        predicts and returns y for the given X
    """
    def predict(self, X):
        y = [] # TODO: might want a numpy array here instead or pd dataframe?
        if self.tree_root = None:
            pass #raise Error or just invalid value?
        else:
            for X_i in X:
                y.append(self.tree_root.get_prediction(X_i))
            return y
                
    
    
    class TreeNode():
        
        def __init__(self, num_children=2):
            if num_children > 0:
                self.children = [None for i in range(num_children)]
            else:
                self.children = None
        
        def get_prediction(self, X_i):
            pass
        
    class InnerNode(TreeNode):
        
        """
        test should be a function that takes a vector/list and based on that returns a non-negative integer < num_children
            test should be set during the training step
        """
        def __init__(self, test, num_children=2):
            super().__init__(num_children=num_children)
            self.test = test
        
        def get_prediction(self, X_i):
            #use test to get the prediction value of one of the child nodes
            # something like: return self.children[test(X_i)].get_prediction(X_i)
            return self.children[self.test(X_i)].get_prediction(X_i)
        
    class LeafNode(TreeNode):
        
        """
            prediction value should either get set to the value of the average of all values contained in this leaf node
        """
        def __init__(self, prediction_value):
            super().__init__(num_children=0)
            self.prediction_value = prediction_value
        
        def get_prediction(self, X_i):
            #return ´own prediction value
            return self.prediction_value


def main():
    pass

if __name__ == '__main__':
    main()