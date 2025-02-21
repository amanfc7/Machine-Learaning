import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

import sys

from data_sets_util import load_ds

# TODO: 5+ ML algs
# 1. MLP
# 2. DT and/or Random Forest
# 3. The GPC?/TBD
# 4. TBD
# 5. TBD

def optimze(X_train, y_train, X_test, y_test, init_T=1000):
    curr_best = None
    T = init_T
    t = 0
    
    classifier_list = [MLPClassifier, RandomForestClassifier]
    
    classifier_0_hp_array = [ # for MLP
        ('identity', 'logistic', 'tanh', 'relu'), #"activation": 
        ('lbfgs', 'sgd', 'adam'), # "solver": 
        ((15,2), (100,), (15,15), (20,3), (50, 50), (20, 20, 20), (100,100,100)), # "hidden_layer_sizes": --> maybe make that 'continious'
        np.logspace(-10, 4, 15), # "alpha": --> continous?
        ]
    classifier_1_hp_array = [ # for TBD
        [],
        ]
    classifier_2_hp_array = [ # for TBD
        [],
        ]
    # classifier_3_hp_array = [ # for TBD
    #     [],
    #     ]
    # classifier_4_hp_array = [ # for TBD
    #     [],
    #     ]

    all_classifiers_array = [classifier_0_hp_array, classifier_1_hp_array, classifier_2_hp_array]
    
    # select random initial solution
    #TODO
    max_hp_number = max([len(classifier_array) for classifier_array in all_classifiers_array])
    init_solution_vect = np.random.rand(1+max_hp_number)
    selected_classifier_index = int(len(all_classifiers_array) * init_solution_vect[0])
    selected_classifier_array = all_classifiers_array[selected_classifier_index]
    # for i, hyperparameter_space in enumerate(selected_classifier_array):




    
    

#     Prozeduresimulated annealing
# begin
#   t=0
#   Intialize T #(initial temperature)
#   select a current solution vc at random
#   evaluate vc
#   repeat
#       repeat
#           select a new solution vn in the neighborhood of vc
#           if eval(vc) < eval(vn) then vc=vn
#           else if random[0,1) e^((eval(v_n)-eval(v_c))/T)then vc=vn
#       until(termination-condition)
#       T=g(T,t) #(do cooling)
#       t=t+1
#   until(halting-criterion)
# end

#idea: encode algs in vectors of length 1+[max hyperparameters to tone over all algs], where the first component indicates the alg and the others indicate teh hyperparameter. If chosen alg has less than max hyperparameters, ignore excess 
#       if some hyperparameters have higher ranges, will probably need to round for lower --> just pick values from 0 to 1 and then scale to range!

#TODO: implement and test different cooling functions
"""
    Return a lowered temperature
"""
def cool_down(T, t):
    return T - 1

"""
    Evaluates the goodness of a solution
"""
def eval_solution(solution):
    pass


"""
    Call the program with an int parameter to optimize for the data set specified by it
    1 - 
    2 - 
    3 -
    4 -
"""
def main():
    ds_to_load = 1

    if len(sys.argv) > 1:
        try:
            ds_to_load = int(sys.argv[1])
        except ValueError:
            print("Could not load specified data set. Ensure the parameter is on of the integers 1-4")


    (X_train, y_train, X_test, y_test) = load_ds(ds_to_load)
    optimze(X_train, y_train, X_test, y_test)




if __name__ == '__main__':
    main()