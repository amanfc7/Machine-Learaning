import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

# TODO: 5+ ML algs
# 1. MLP
# 2. DT and/or Random Forest
# 3. The GPC?/TBD
# 4. TBD
# 5. TBD

def optimze(X_train, y_train, X_test, y_test, init_T=1000):
    curr_best = None
    T = init_T
    
    classifier_list = [MLPClassifier, RandomForestClassifier]
    
    classifier_0_hp_array = np.array([ # for MLP
        ('identity', 'logistic', 'tanh', 'relu'), #"activation": 
        ('lbfgs', 'sgd', 'adam'), # "solver": 
        ((15,2), (100,), (15,15), (20,3), (50, 50), (20, 20, 20), (100,100,100)), # "hidden_layer_sizes": --> maybe make that 'continious'
        np.logspace(-10, 4, 15), # "alpha": --> continous?
        ])
    classifier_1_hp_array = np.array([ # for TBD
        [],
        ])
    classifier_2_hp_array = np.array([ # for TBD
        [],
        ])
    classifier_3_hp_array = np.array([ # for TBD
        [],
        ])
    classifier_4_hp_array = np.array([ # for TBD
        [],
        ])
    
    

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

def main():
    pass
    # load()
    # optimze(X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    main()