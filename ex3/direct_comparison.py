import sys

from data_sets_util import load_ds
import custom_sim_ann
import TPOT_setup
import pycaret_setup
from pycaret.classification import *

def compare_methods_for_ds(data_set_index):
    X_train, y_train, X_test, y_test = load_ds(data_set_index)

    # TODO: run own implementation, TPOT and auto-sklearn on data sets, analyse & plot results
    
    # TPOT
    optimizer = TPOT_setup.optimize(X_train, y_train, X_test, y_test, ds_index=data_set_index)

    # 2nd pycaret. 
    # if a pipeline file already exists, use it
    try:
        best = load_model('best_pipeline_for_ds_'+str(data_set_index))
    except FileNotFoundError:
        best = pycaret_setup.optimize(X_train, y_train, X_test, y_test, ds_index=data_set_index)
    

    # custom
    found_clf = custom_sim_ann.optimize(X_train, y_train, X_test, y_test,ds_index=data_set_index)
    # or just import from log file if it already exists and has finished - do it manually for now

    #TODO:# comparison plots
    


def main():
    ds_to_load = 1

    if len(sys.argv) > 1:
        try:
            ds_to_load = int(sys.argv[1])
        except ValueError:
            print("Could not load specified data set. Ensure the parameter is one of the integers 1-4")

    compare_methods_for_ds(ds_to_load)


if __name__ == '__main__':
    main()
