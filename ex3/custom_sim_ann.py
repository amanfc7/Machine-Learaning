import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

import sys, time

from data_sets_util import load_ds

# TODO: 5+ ML algs
# 1. MLP
# 2. DT and/or Random Forest
# 3. The GPC?/TBD
# 4. TBD
# 5. TBD



def optimze(X_train, y_train, X_test, y_test, init_T=1000):
    start_time = time.time()
    curr_best = None
    curr_best_score = 0.
    T = init_T
    t = 0
    
    classifier_0 =  [
        MLPClassifier,
        { # hyperparameter dict
            "activation": ('identity', 'logistic', 'tanh', 'relu'),
            "solver": ('lbfgs', 'sgd', 'adam'),
            "hidden_layer_sizes": ((15,2), (100,), (15,15), (20,3), (50, 50), (20, 20, 20), (100,100,100)),
            "alpha": np.logspace(-10, 4, 15),
            # "max_iter": (200, 300),
        }
    ]
    # classifier_0_hp_array = [ # for MLP
    #     ('identity', 'logistic', 'tanh', 'relu'), #"activation": 
    #     ('lbfgs', 'sgd', 'adam'), # "solver": 
    #     ((15,2), (100,), (15,15), (20,3), (50, 50), (20, 20, 20), (100,100,100)), # "hidden_layer_sizes": --> maybe make that 'continious'
    #     np.logspace(-10, 4, 15), # "alpha": --> continous?
    #     ]
    #TODO
    classifier_1 =  [
        RandomForestClassifier,
        { # hyperparameter dict
            
        }
    ]
    classifier_2 =  [
        RandomForestClassifier,
        { # hyperparameter dict
            
        }
    ]
    #TODO
    classifier_3 =  [
        RandomForestClassifier,
        { # hyperparameter dict
            
        }
    ]
    #TODO
    classifier_4 =  [
        RandomForestClassifier,
        { # hyperparameter dict
            
        }
    ]

    all_classifiers_array = [classifier_0, classifier_1, classifier_2, classifier_3, classifier_4]
    
    # select random initial solution
    #TODO: seems fine, check again
    max_hp_number = max([len(classifier[1].keys()) for classifier in all_classifiers_array])
    init_solution_vect = np.random.rand(1+max_hp_number)
    
    clf = solution_vect_to_clf(init_solution_vect, all_classifiers_array)
    curr_best_score = eval_solution(clf, X_train, y_train, X_test, y_test)
    curr_score = curr_best_score

    current_solution = init_solution_vect
    current_best = init_solution_vect

    # now loop until stop
    # stopping criterion is that one hour has passed (and/or perfect accuracy achieved?)


    while (halting_criterion(start_time)): #one hour has not yet passed
        i = 0
        while (termination_codition(i)):
            i = i + 1
            new_solution = select_neighbor(current_solution, all_classifiers_array, T)
            new_score = eval_solution(solution_vect_to_clf(new_solution, all_classifiers_array), X_train, y_train, X_test, y_test)
            if curr_score < new_score:
                current_solution = new_solution
                curr_score = new_score
                if curr_best_score < curr_score:
                    current_best = current_solution
                    curr_best_score = curr_score
            else:
                if np.random.rand(1) < np.exp((new_score - curr_score) / T):
                    current_solution = new_solution
                    curr_score = new_score


        T = cool_down(T, t)
        t = t + 1
        # print(t)

    

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

# should temperature restrict the size of the neighborhood or just affect the chance with which a worse solution gets kept? (easier) 


"""
    Returns a solution in the neighborhood of the current solution
"""
def select_neighbor(solution, all_classifiers_array, T):
    return solution

#TODO: implement and test different cooling functions
"""
    Return a lowered temperature
"""
def cool_down(T, t):
    reduction_factor = 0.8
    return T * reduction_factor
    # return T - 1

"""
    Termination condition for one time step
"""
def termination_codition(i):
    return i < 10

"""
    Halting criterion for the whole algorithm
    Returns True while start_time is at most 1 hour before the current time, otw return False
"""
def halting_criterion(start_time):
    if time.time() - start_time < (1 * 60 * 60): #less than one hour has passed
        return True
    return False

"""
    Evaluates the goodness of a solution
"""
def eval_solution(solution_clf, X_train, y_train, X_test, y_test):
    solution_clf.fit(X_train, y_train)
    return solution_clf.score(X_test, y_test)

#TODO: double check if this makes sense
"""
    turns a solution vector into a classifier object and returns it
"""
def solution_vect_to_clf(solution_vect, solution_space):
    selected_classifier_index = int(len(solution_space) * solution_vect[0])
    selected_classifier = solution_space[selected_classifier_index]
    chosen_hyperparameter_dict = {}
    for i, key in enumerate(selected_classifier[1].keys()):
        possible_values = selected_classifier[1][key]
        solution_value_index = int(len(possible_values) * solution_vect[i+1])
        chosen_hyperparameter_dict[key] = possible_values[solution_value_index]

    
    clf = selected_classifier[0](**chosen_hyperparameter_dict)
    return clf


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