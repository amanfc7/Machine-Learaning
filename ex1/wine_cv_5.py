from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, DotProduct, RationalQuadratic, ConstantKernel as C
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import sys
from ds_load_util import load_dataset  # Ensure this utility is correctly implemented


def main():
    search = False
    for arg in sys.argv:
        if arg == '-s':
            search = True
    train_model(search)


def train_model(do_search=False, scaler_no=2):
    # Select scaler
    if scaler_no == 1:
        scaler = preprocessing.StandardScaler()
    elif scaler_no == 2:
        scaler = preprocessing.MinMaxScaler()
    elif scaler_no == 3:
        scaler = preprocessing.RobustScaler()
    else:
        scaler = None

    # 1. Load dataset
    X_train, X_test, y_train, y_test = load_dataset(
        'wine',
        preprocess=True,
        scaler=scaler,
    )

    # 2. Define kernel for GPC
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))

    if do_search:
        # 3. Hyperparameter tuning with GridSearchCV
        parameters = {
            "kernel": [
                C(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)),
                C(0.5, (1e-3, 1e2)) * RBF(length_scale=0.5, length_scale_bounds=(1e-3, 1e1)),
                C(1.0, (1e-2, 1e2)) * Matern(length_scale=1.5, length_scale_bounds=(1e-2, 1e2)),
                C(1.0, (1e-3, 1e2)) * Matern(length_scale=0.5, length_scale_bounds=(1e-3, 1e1)),
                C(1.0, (1e-2, 1e2)) * DotProduct(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)),
                C(1.0, (1e-3, 1e2)) * DotProduct(length_scale=0.5, length_scale_bounds=(1e-3, 1e1)),
                C(1.0, (1e-2, 1e2)) * RationalQuadratic(length_scale=0.5, length_scale_bounds=(1e-3, 1e1))
            ],
            "optimizer": [None, "fmin_l_bfgs_b"],
            "max_iter_predict": [100, 300],
            "n_restarts_optimizer": [0, 2],
        }

        search = GridSearchCV(
            estimator=GaussianProcessClassifier(kernel=kernel, random_state=1),
            param_grid=parameters,
            cv=5,
            n_jobs=4,
            verbose=1,
        )

        print("Performing grid search...")
        from time import time

        t0 = time()
        search.fit(X_train, y_train)
        print(f"Done in {time() - t0:.3f}s")

        print("Best parameters combination found:")
        best_parameters = search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print(f"{param_name}: {best_parameters[param_name]}")

        clf = search.best_estimator_

        test_accuracy = search.score(X_test, y_test)
        print(
            "Accuracy of the best parameters using the inner CV of "
            f"the grid search: {search.best_score_:.3f}"
        )
        print(f"Accuracy on test set: {test_accuracy:.3f}")

    else:
        # 4. Train and evaluate a GPC with default kernel
        clf = GaussianProcessClassifier(kernel=kernel, random_state=1)
        clf.fit(X_train, y_train)

        # Accuracy from holdout test set
        print("Test Accuracy:", clf.score(X_test, y_test))

        # Cross-validation
        scores = cross_val_score(clf, X_train, y_train, cv=10)
        print("CV scores:")
        print(scores)
        print("CV: %0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

    print(f"Scaler number: {scaler_no}, Kernel: {kernel}")


if __name__ == '__main__':
    main()
