import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

import sys, time

from data_sets_util import load_ds

# 5+ ML algs
# 1. MLP
# 2. KNC 
# 3. SVC
# 4. DTC
# 5. GBC

classifier_1 =  [
    MLPClassifier,
    { # hyperparameter dict
        "activation": ('identity', 'logistic', 'tanh', 'relu'),
        "solver": ('lbfgs', 'sgd', 'adam'),
        "hidden_layer_sizes": ((15,2), (100,), (15,15), (20,3), (50, 50), (20, 20, 20)), # --> maybe make that 'continious'
        "alpha": np.logspace(-10, 4, 15), # --> maybe make that 'continious'
        # "max_iter": (200, 300),
    }
]

classifier_2 =  [
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

classifier_3 =  [
    SVC,
    { # hyperparameter dict
        "C": np.logspace(-3, 3, 10),  
        # "kernel": ("linear", "poly", "rbf", "sigmoid"), 
        "kernel": ("poly", "rbf", "sigmoid"), 
        "degree": tuple(range(2, 6)),  
        "gamma": ("scale", "auto"), 
        "coef0": np.linspace(0, 1, 5),  
        "shrinking": (True, False), 
        # "probability": (True, False), # setting to True has some problems
        "class_weight": (None, "balanced"), 
        "cache_size": [1000] #for faster run time
    }
]


classifier_4 =  [
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

classifier_5 =  [
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
        # "loss": ["log_loss", "exponential"],  #TODO: exponential only works for binary classification, maybe remove this hyperparameter? Or only allow it dynamically for BinClassProblems
        "validation_fraction": [0.1, 0.2, 0.3],  
        "n_iter_no_change": [None, 10, 20],
        "tol": [1e-4, 1e-3, 1e-2], 
    }
]

# classifier_6 = [
#     RandomForestClassifier,
#     {

#     }
# ]

all_classifiers_array = [classifier_1, classifier_2, classifier_3, classifier_4, classifier_5]

"""

    verbosity has three levels: 0 - don't print stuff, 1 - only print what is also logged, 2 - also print each new t once

    init_T: the initial temperature
    min_T: minimum temperature
    reset_T: temperature to reset to once T gets too low
    reduction_factor: factor by which to lower the temperature
    max_it_per_t: how many times a neighbor should be picked per t
    c_1: a 'weight' for staying with the same classifier
    c_2: how much the temperature affects the chance of selecting a different classifier
"""
def optimize(X_train, y_train, X_test, y_test, rng_seed=None, ds_index=0, verbosity=1,
    init_T=150, min_T=10, reset_T=100, reduction_factor=0.9995, max_it_per_t=10, c_1 = 3, c_2 = 100):
    log_file_name = "custom_sim_ann_for_ds_" + str(ds_index) + ".log"
    rng = np.random.default_rng(rng_seed)
    start_time = time.time()
    

    max_hp_number = max([len(classifier[1].keys()) for classifier in all_classifiers_array])
    
    # time stamp = 0
    t = 0
    # Intialize T #(initial temperature)
    T = init_T 
           

   # select random initial current solution v_c
    init_solution_vect = rng.random(1+max_hp_number)
    clf = solution_vect_to_clf(init_solution_vect)
    # Evalutate v_c
    curr_best_score = eval_solution_adjusted(clf, X_train, y_train, X_test, y_test)
    curr_score = curr_best_score

    current_solution = init_solution_vect
    current_best = init_solution_vect

    if ds_index != 0:
        with open(log_file_name, "a") as myfile: # should "a" be "w" instead to overwrite the log for each new run?
            myfile.write("Started simulated annealing optimization for data set with index  %d\n" % ds_index)
            myfile.write("Parameters: init_T=%d, min_T=%d, reset_T=%d, reduction_factor=%f, max_it_per_t=%d, c_1 = %d, c_2 = %d\n\n" % (init_T, min_T, reset_T, reduction_factor, max_it_per_t, c_1, c_2))

    
    last_log_time = start_time

    # now loop until th halting criterion is reached
    while (halting_criterion(start_time)): #one hour has not yet passed
        i = 0
        # loop until the termination condition for the current time step has been reached
        while (termination_condition(i, T, max_it_per_t)):
            i = i + 1
            # select a new solution v_n in the neighborhood of v_c ...
            new_solution = select_neighbor(current_solution, T, rng, c_1, c_2)
            # ... and evalute it
            clf = solution_vect_to_clf(new_solution)
            if verbosity > 1: 
                print(f'Training an instance of the {str(type(clf)).split(".")[-1][:-2]}\n')
                if type(clf) == SVC: print(clf.get_params())
            new_score = eval_solution_adjusted(clf, X_train, y_train, X_test, y_test)
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


        T = cool_down(T, t, min_T, reset_T, reduction_factor)
        t = t + 1
        if verbosity > 1: print(t)

        #report regularly about what is currently going on, also log it
        if time.time() - last_log_time >= 60:
            last_log_time = time.time()
            clf = solution_vect_to_clf(current_best)
            mins_since_start = (time.time() - start_time) / 60.
            with open(log_file_name, "a") as myfile:
                myfile.write("t = %s, T = %f\n" % (t, T))
                myfile.write(f'Current best score: {curr_best_score/100:0.5f} for the {str(type(clf)).split(".")[-1][:-2]}\n')
                myfile.write("Selected parameters:\n")
                myfile.write(str(clf.get_params()))
                myfile.write("\nElapsed mins since start: %.2f" % (mins_since_start))
                myfile.write("\n\n")
                # myfile.write(str(current_best))
            if verbosity > 0:
                print("t = %s, T = %f" % (t, T))
                print(f'Current best score: {curr_best_score/100:0.5f} for the {str(type(clf)).split(".")[-1][:-2]}')
                print("Selected parameters:")
                print(clf.get_params())
                print("Elapsed mins since start: %.2f" % (mins_since_start))

    clf = solution_vect_to_clf(current_best)
    print("Finished with the following result:")
    print(f'Best score: {curr_best_score/100:0.5f} for the {str(type(clf)).split(".")[-1][:-2]}')
    print("Selected parameters:")
    print(clf.get_params())
    with open(log_file_name, "a") as myfile:
        myfile.write("Finished with the following result at t = %s\n" % t)
        myfile.write(f'Best score: {curr_best_score/100:0.5f} for the {str(type(clf)).split(".")[-1][:-2]}\n')
        myfile.write("Selected parameters:\n")
        myfile.write(str(clf.get_params()))
        myfile.write("\nTotal run time in mins: %.2f" % (time.time() - start_time))
        myfile.write("\nVector for best solution:\n")
        myfile.write(str(current_best))


    # at the end, return the best classifier found
    return clf


#idea: encode algs in vectors of length 1+[max hyperparameters to tone over all algs], where the first component indicates the alg and the others indicate the hyperparameter. If chosen alg has less than max hyperparameters, ignore excess 
#       if some hyperparameters have higher ranges, will probably need to round for lower --> just pick values from 0 to 1 and then scale to range!



"""
    Returns a solution in the neighborhood of the current solution

    c_1: a 'weight' for staying with the same classifier
    c_2: how much the temperature affects the chance of selecting a different classifier
"""
def select_neighbor(solution, T, rng, c_1, c_2, centering=True):
    selected_classifier_index = int(len(all_classifiers_array) * solution[0])
    new_solution = solution.copy()

    # select a (possibly new) classifier, other classifiers should prbly be 'further away'
    possible_clf_ind_list = [selected_classifier_index] * (c_1 + int(c_2/T))
    if selected_classifier_index == 0: #we are at left border, add right neighbor twice
        possible_clf_ind_list.append(1)
    else:
        possible_clf_ind_list.append(selected_classifier_index-1)
    if selected_classifier_index == (len(all_classifiers_array)-1): #we are at the right border, add left neighbor twice
        possible_clf_ind_list.append(selected_classifier_index-1)
    else:
        possible_clf_ind_list.append(selected_classifier_index+1)

    new_clf_index = rng.choice(possible_clf_ind_list)
    # transform back into our 0-1 range with a little extra added to fall right in the middle of the range for the index
    if centering:
        new_solution[0] = new_clf_index / len(all_classifiers_array) + 1 / (2 * len(all_classifiers_array))
    else:
        new_solution[0] = new_clf_index / len(all_classifiers_array) # w/o centering, might cause unwanted parameter shifts if switching back and forth between clfs

    # update the remaining part of the solution vector where appropriate
    new_classifier = all_classifiers_array[new_clf_index]

    for i, key in enumerate(new_classifier[1].keys()): # keys should always be in same order since solution space doesn't change
        possible_values = new_classifier[1][key]

        if int(0.1 * T) > 1: #even at T = 10 we want at least one step, possibly more at higher temperatures
            choices_steps_to_add = [i for i in range(int(0.1 * T)+1)]
        else:
            choices_steps_to_add = [0, 1]
        steps_to_add = rng.choice(choices_steps_to_add)
        steps_to_add = steps_to_add * rng.choice([-1, 1]) # subtract or add

        value_to_add = steps_to_add / len(possible_values) # shift in to our range

        new_solution_value = solution[i+1] + value_to_add
        # value to small, should be at least 0
        if new_solution_value < 0:
            new_solution_value = 0
        # value to large, need largest valid value under 1
        if new_solution_value >= 1:
            if centering:
                new_solution_value = 1 - 1 / (2 * len(possible_values))
            else:
                new_solution_value = 1 - 1 / len(possible_values) # w/o centering
        new_solution[i+1] = new_solution_value

    return new_solution


"""
    Return a lowered temperature

    T: current temperature
    t: current time stamp
    min_T: minimum temperature
    reset_T: temperature to reset to once T gets too low
    reduction_factor: factor by which to lower the temperature
"""
def cool_down(T, t, min_T, reset_T, reduction_factor):
    new_T = T * reduction_factor
    if new_T < min_T:
        new_T = reset_T
    return new_T

"""
    Termination condition for one time step.
    Simply returns false once i is large enough

    max_it_per_t: how many times a neighbor should be picked for each t
"""
def termination_condition(i, T, max_it_per_t):
    max_iterations = max_it_per_t
    return i < max_iterations

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
    Trains the classifier on the provided data and evalutes it, then scales by 100 the result for a wider range
"""
def eval_solution_adjusted(solution_clf, X_train, y_train, X_test, y_test):
    return eval_solution(solution_clf, X_train, y_train, X_test, y_test) * 100

"""
    Turns a solution vector into a classifier object and returns it
"""
def solution_vect_to_clf(solution_vect):
    solution_space=all_classifiers_array
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
    1 - Sick
    2 - Congressional voting
    3 - Waveform
    4 - Wine
"""
def main():
    ds_to_load = 1

    if len(sys.argv) > 1:
        try:
            ds_to_load = int(sys.argv[1])
        except ValueError:
            print("Could not load specified data set. Ensure the parameter is one of the integers 1-4")

    X_train, y_train, X_test, y_test = load_ds(ds_to_load)
    optimize(X_train, y_train, X_test, y_test, ds_index=ds_to_load)


if __name__ == '__main__':
    main()