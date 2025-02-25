import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

import sys, time

from data_sets_util import load_ds

# TODO: 5+ ML algs
# 0. MLP
# 1. KNC 
# 2. SVC
# 3. DTC
# 4. GBC



def optimize(X_train, y_train, X_test, y_test, init_T=1000, rng_seed=None):
    rng = np.random.default_rng(rng_seed)
    start_time = time.time()
    curr_best = None
    curr_best_score = 0.
    
    classifier_0 =  [
        MLPClassifier,
        { # hyperparameter dict
            "activation": ('identity', 'logistic', 'tanh', 'relu'),
            "solver": ('lbfgs', 'sgd', 'adam'),
            "hidden_layer_sizes": ((15,2), (100,), (15,15), (20,3), (50, 50), (20, 20, 20), (100,100,100)), # --> maybe make that 'continious'
            "alpha": np.logspace(-10, 4, 15), # --> maybe make that 'continious'
            # "max_iter": (200, 300),
        }
    ]

    classifier_1 =  [
        KNeighborsClassifier,
        { # hyperparameter dict
            "n_neighbors": tuple(range(1, 21)), 
            "weights": ("uniform", "distance"), 
            "algorithm": ("auto", "ball_tree", "kd_tree", "brute"), 
            "leaf_size": tuple(range(10, 51, 5)),
            "p": (1, 2),
            "metric": ("euclidean", "manhattan", "chebyshev", "minkowski"), 
        }
    ]

    classifier_2 =  [
        SVC,
        { # hyperparameter dict
            "C": np.logspace(-3, 3, 10),  
            "kernel": ("linear", "poly", "rbf", "sigmoid"), 
            "degree": tuple(range(2, 6)),  
            "gamma": ("scale", "auto"), 
            "coef0": np.linspace(0, 1, 5),  
            "shrinking": (True, False), 
            "probability": (True, False), 
            "class_weight": (None, "balanced"), 
        }
    ]


    classifier_3 =  [
        DecisionTreeClassifier,
        { # hyperparameter dict
            "criterion": ("gini", "entropy", "log_loss"),  
            "splitter": ("best", "random"), 
            "max_depth": tuple(range(3, 21, 3)), 
            "min_samples_split": tuple(range(2, 21)), 
            "min_samples_leaf": tuple(range(1, 21)), 
            "max_features": ("sqrt", "log2", None), 
            "class_weight": (None, "balanced"),
        }
    ]

    classifier_4 =  [
        GradientBoostingClassifier,
        { # hyperparameter dict
            "n_estimators": [50, 100, 200, 300],  
            "learning_rate": np.logspace(-3, 0, 4), 
            "max_depth": [3, 5, 7, 10],  
            "min_samples_split": [2, 5, 10],  
            "min_samples_leaf": [1, 2, 5], 
            "subsample": [0.5, 0.7, 1.0], 
            "max_features": [None, "sqrt", "log2"],
            "warm_start": [True, False],
            "loss": ["deviance", "exponential"],  
            "validation_fraction": [0.1, 0.2, 0.3],  
            "n_iter_no_change": [None, 10, 20],
            "tol": [1e-4, 1e-3, 1e-2], 
        }
    ]

    all_classifiers_array = [classifier_0, classifier_1, classifier_2, classifier_3, classifier_4]
    max_hp_number = max([len(classifier[1].keys()) for classifier in all_classifiers_array])
    
    # time stamp = 0
    t = 0
    # Intialize T #(initial temperature)
    T = init_T 
           

    # select random initial current solution v_c
    #TODO: seems fine, check again
    init_solution_vect = rng.random(1+max_hp_number)
    clf = solution_vect_to_clf(init_solution_vect, all_classifiers_array)
    # Evalutate v_c
    curr_best_score = eval_solution(clf, X_train, y_train, X_test, y_test)
    curr_score = curr_best_score

    current_solution = init_solution_vect
    current_best = init_solution_vect

    # now loop until th halting criterion is reached
    while (halting_criterion(start_time)): #one hour has not yet passed
        i = 0
        # loop until the termination condition for the current time step has been reached
        while (termination_condition(i)):
            i = i + 1
            # select a new solution v_n in the neighborhood of v_c ...
            new_solution = select_neighbor(current_solution, all_classifiers_array, T)
            # ... and evalute it
            new_score = eval_solution(solution_vect_to_clf(new_solution, all_classifiers_array), X_train, y_train, X_test, y_test)
            # if it's better than v_c, update v_c
            if curr_score < new_score:
                current_solution = new_solution
                curr_score = new_score
                if curr_best_score < curr_score:
                    current_best = current_solution
                    curr_best_score = curr_score
            # even if it's not better, update with a certain probability
            else:
                if rng.random() < np.exp((new_score - curr_score) / T):
                    current_solution = new_solution
                    curr_score = new_score


        T = cool_down(T, t)
        t = t + 1
        # print(t)

        #TODO: we might want to report regularly about what is currently going on

    #TODO:
    # we want to return/do something with the best found solution
    print(current_best)
    print(curr_best_score)


#idea: encode algs in vectors of length 1+[max hyperparameters to tone over all algs], where the first component indicates the alg and the others indicate the hyperparameter. If chosen alg has less than max hyperparameters, ignore excess 
#       if some hyperparameters have higher ranges, will probably need to round for lower --> just pick values from 0 to 1 and then scale to range!

# should temperature restrict the size of the neighborhood or just affect the chance with which a worse solution gets kept? (easier) 

#TODO: implement/test different neighborhood selectors
"""
    Returns a solution in the neighborhood of the current solution
"""
def select_neighbor(solution, all_classifiers_array, T):
    # other classifiers should prbly be 'further away'
    return solution

#TODO: implement and test different cooling functions
"""
    Return a lowered temperature
"""
def cool_down(T, t):
    reduction_factor = 0.8
    return T * reduction_factor
    # return T - 1

#TODO: implement/test different conditions
"""
    Termination condition for one time step
"""
def termination_condition(i):
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
    Trains the classifier on the provided data and evalutes it
"""
def eval_solution(solution_clf, X_train, y_train, X_test, y_test):
    solution_clf.fit(X_train, y_train)
    return solution_clf.score(X_test, y_test)

"""
    Turns a solution vector into a classifier object and returns it
"""
def solution_vect_to_clf(solution_vect, solution_space):
    selected_classifier_index = int(len(solution_space) * solution_vect[0])
    selected_classifier = solution_space[selected_classifier_index]
    chosen_hyperparameter_dict = {}
    for i, key in enumerate(selected_classifier[1].keys()): # keys should always be in same order since solution space doesn't change
        possible_values = selected_classifier[1][key]
        solution_value_index = int(len(possible_values) * solution_vect[i+1])
        chosen_hyperparameter_dict[key] = possible_values[solution_value_index]

    clf = selected_classifier[0](**chosen_hyperparameter_dict) # build selected classifier by passing hyperparameters as a dict
    return clf


"""
    Call the program with an int parameter to optimize for the data set specified by it
    0 -
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
            print("Could not load specified data set. Ensure the parameter is one of the integers 0-4")

    X_train, y_train, X_test, y_test = load_ds(ds_to_load)
    optimize(X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    main()
