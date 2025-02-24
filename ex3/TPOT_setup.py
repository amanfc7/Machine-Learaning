
# simple setup for utilizing TPOT. might just be done directly in comparison, maybe
# TODO: set up so it works with our/specified data sets
#   right now its just the basic initial test

# below is example use
from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import sys
from data_sets_util import load_ds


# TODO: adjust so it can better/more easily be used for comparison
def optimize(X_train, X_test, y_train, y_test):
    pipeline_optimizer = TPOTClassifier(generations=5, population_size=20, cv=5,
                                        random_state=42, verbosity=2)
    pipeline_optimizer.fit(X_train, y_train)
    print(pipeline_optimizer.score(X_test, y_test))
    pipeline_optimizer.export('tpot_exported_pipeline.py')


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