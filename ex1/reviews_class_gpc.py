from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct, ConstantKernel as C
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import sys
from ds_load_util import load_dataset  # Ensure this utility is correctly implemented


def main():
    search = True
    for arg in sys.argv:
        if arg == '-s':
            search = True
    train_model(search)


def train_model(do_gridsearch=False, scaler_no=2):
    # Select scaler
    if scaler_no == 1:
        scaler = preprocessing.StandardScaler()
    elif scaler_no == 2:
        scaler = preprocessing.MinMaxScaler()
    elif scaler_no == 3:
        scaler = preprocessing.RobustScaler()
    elif scaler_no == 4:
        scaler=preprocessing.MaxAbsScaler()
    else:
        scaler = None

    # 1. Load dataset
    X_train, X_test, y_train, y_test = load_dataset(
        'reviews',
        preprocess=True,
        #encoder=preprocessing.OrdinalEncoder(),
        # encoder=preprocessing.OneHotEncoder(),
        imputer=SimpleImputer(strategy="constant", fill_value=-1),
        #imputer=SimpleImputer(),
        scaler=scaler,
        # ("normalizer", preprocessing.Normalizer()),
    )

    # # 2. data exploration and preprocessing
    # enc = Pipeline(steps=[
    #     ("encoder", preprocessing.OrdinalEncoder()),
    #     # ("encoder", preprocessing.OneHotEncoder()),
    #     ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
    #     ("scaler", preprocessing.StandardScaler()),
    #     # ("scaler", preprocessing.RobustScaler()),
    #     # ("scaler", preprocessing.MaxAbsScaler()),
    #     # ("scaler", preprocessing.MinMaxScaler()),
    #     # ("scaler", preprocessing.StandardScaler(with_mean=False)),
    #     # ("normalizer", preprocessing.Normalizer()),
    # ])


    # 2. gridsearch
    if do_gridsearch:
        parameters = {
            "kernel": [
                C(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)),
                C(0.5, (1e-3, 1e2)) * RBF(length_scale=0.5, length_scale_bounds=(1e-3, 1e1)),
                C(1.0, (1e-2, 1e2)) * Matern(length_scale=1.5, length_scale_bounds=(1e-2, 1e2)),
                C(1.0, (1e-3, 1e2)) * Matern(length_scale=0.5, length_scale_bounds=(1e-3, 1e1)),
                C(1.0, (1e-2, 1e2)) * DotProduct(),
                C(1.0, (1e-3, 1e2)) * DotProduct(),
                C(1.0, (1e-2, 1e2)) * RationalQuadratic(length_scale=0.5, length_scale_bounds=(1e-3, 1e1))
            ],
            "optimizer": [None, "fmin_l_bfgs_b"],
            "max_iter_predict": [100, 300],
           # "length_bond_scales" : 
            "n_restarts_optimizer": [0, 2],
        }

         

        grid_search = GridSearchCV(
            GaussianProcessClassifier(random_state=1),
            parameters,
            cv=5,
            n_jobs=2,
            #verbose=1,
    )
        print("Performing grid search...")
        print("Hyperparameters to be evaluated:")
        # pprint(parameter_grid)
    
        from time import time

        t0 = time()
        grid_search.fit(X_train, y_train)
        print(f"Done in {time() - t0:.3f}s")

        print("Best parameters combination found:")
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print(f"{param_name}: {best_parameters[param_name]}")

        clf = grid_search.best_estimator_

        test_accuracy = grid_search.score(X_test, y_test)
        print(
            "Accuracy of the best parameters using the inner CV of "
            f"the grid search: {grid_search.best_score_:.3f}"
        )
        print(f"Accuracy on test set: {test_accuracy:.3f}")

    else:
        # Choose model based on scaler
        if scaler_no == 1:  # better with StandardScaler
            clf = GaussianProcessClassifier()
        elif scaler_no == 2:  # better with MinMaxScaler
            clf = GaussianProcessClassifier()
        elif scaler_no == 3:  # better with RobustScaler
            clf = GaussianProcessClassifier()

        clf.fit(X_train, y_train)
        # accuracy & precision, false positives, false negatives
        
        scores = cross_val_score(clf, X_train, y_train, cv=10)

        print(clf.score(X_test, y_test))
        print("accurancy from holdout\n")

        #crossvalidation
        clf = GaussianProcessClassifier()
        # scores = cross_val_score(clf, X, y, cv=10)
        print(scores)
        print("CV: %0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

    print("Scaler number: %d" % scaler_no)
    
    return (clf, X_test, y_test)


if __name__ == '__main__':
    main()
