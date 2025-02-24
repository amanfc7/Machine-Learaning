
# simple setup for utilizing another AUTO ML  implementation (pycaret). might just be done directly in comparison, maybe


from pycaret.classification import ClassificationExperiment

import sys
from data_sets_util import load_ds


# TODO: adjust so it can better/more easily be used for comparison
def optimize(X_train, y_train, X_test, y_test):

    # pycaret requires X and y to be in one table, with the target column specified. 
    # TODO: Modify below accordingly
    data = X_train
    data_test = X_test

    s = ClassificationExperiment()
    ## s.setup(data, target = 'Purchase', session_id = 123)
    s.setup(data, target = 0, test_data=data_test)

    # model training and selection
    best = s.compare_models()

    # evaluate trained model
    s.evaluate_model(best)

    # predict on hold-out/test set
    pred_holdout = s.predict_model(best)

    # # predict on new data
    # new_data = data.copy().drop('Purchase', axis = 1)
    # predictions = s.predict_model(best, data = new_data)

    # save model
    s.save_model(best, 'best_pipeline')



def main():
    ds_to_load = 1

    if len(sys.argv) > 1:
        try:
            ds_to_load = int(sys.argv[1])
        except ValueError:
            print("Could not load specified data set. Ensure the parameter is one of the integers 1-4")

    X_train, y_train, X_test, y_test = load_ds(ds_to_load)
    optimize(X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    main()