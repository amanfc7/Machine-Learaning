#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#vgl 06


from DT_regressor import DTRegressor
from sklearn.tree import DecisionTreeRegressor

class RandomForestRegressor():
    
    
    """
        use_skl_tree: should the random forest be built from trees from sklearn? For testing/comparsion purposes, default: False
    """
    def __init__(self, use_skl_tree=False, num_trees=100, max_depth=-1):
        if not use_skl_tree:
            self.TreeClass = DTRegressor
        else:
            self.TreeClass = DecisionTreeRegressor
            
        

def main():
    pass



if __name__ == '__main__':
    main()