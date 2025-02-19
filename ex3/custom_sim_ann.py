

# TODO: 5+ ML algs
# 1. MLP
# 2. DT and/or Random Forest
# 3. The GPC?/TBD
# 4. TBD
# 5. TBD

def omptimze(X_train, y_train, X_test, y_test):
    curr_best = None

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