import sys

from data_sets_util import load_ds
import custom_sim_ann
import TPOT_setup
import pycaret_setup

def compare_methods_for_ds(data_set_index):
    X_train, y_train, X_test, y_test = load_ds(data_set_index)

    # TODO: run own implementation, TPOT and auto-sklearn on data sets, analyse & plot results
    # custom
    custom_sim_ann.optimize(X_train, y_train, X_test, y_test)

    # TPOT
    TPOT_setup.optimize(X_train, y_train, X_test, y_test)

    # 2nd

    # plots



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
