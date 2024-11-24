from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn import preprocessing
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


def select_inducing_points(X, y, n_inducing):
    """
    Select inducing points using KMeans clustering or random subsampling.
    """

    # Convert to numpy array if X or y are pandas DataFrames/Series
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        X = X.to_numpy()
    if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
        y = y.to_numpy()

    if n_inducing < len(X):
        kmeans = KMeans(n_clusters=n_inducing, random_state=0)
        cluster_labels = kmeans.fit_predict(X)
        # Select one representative point per cluster
        inducing_indices = [np.where(cluster_labels == i)[0][0] for i in range(n_inducing)]
        return X[inducing_indices], y[inducing_indices]
    else:
        # Use the full dataset if n_inducing >= len(X)
        return X, y

def train_model(do_search=False):
    # Example: Use 500 inducing points
    return train_model_sparse_gpc(n_inducing=300, scaler_no=4)
    
def train_model_sparse_gpc(n_inducing=100, scaler_no=2):
    # Select scaler
    if scaler_no == 1:
        scaler = preprocessing.StandardScaler()
    elif scaler_no == 2:
        scaler = preprocessing.MinMaxScaler()
    elif scaler_no == 3:
        scaler = preprocessing.RobustScaler()
    elif scaler_no == 4:
        scaler = preprocessing.MaxAbsScaler()
    else:
        scaler = None

    # 1. Load dataset
    X_train, X_test, y_train, y_test = load_dataset(
        'reviews',
        preprocess=True,
        imputer=SimpleImputer(strategy="constant", fill_value=-1),
        scaler=scaler,
    )

    # Reset indices if data is in pandas format
    if isinstance(X_train, pd.DataFrame) or isinstance(X_train, pd.Series):
        X_train = X_train.reset_index(drop=True)
    if isinstance(y_train, pd.DataFrame) or isinstance(y_train, pd.Series):
        y_train = y_train.reset_index(drop=True)

    # 2. Select inducing points
    X_inducing, y_inducing = select_inducing_points(X_train, y_train, n_inducing)

    # 3. Define and train Sparse Gaussian Process Classifier
    kernel = RBF(length_scale=1.0)
    # kernel = Matern(length_scale=1.0, nu=1.5)
    clf = GaussianProcessClassifier(kernel=kernel, random_state=1)

    print(f"Training Sparse GPC with {n_inducing} inducing points...")
    clf.fit(X_inducing, y_inducing)

    # 4. Evaluate
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    print(f"Accuracy on training set: {train_score:.3f}")
    print(f"Accuracy on test set: {test_score:.3f}")

    # Cross-validation for inducing points
    scores = cross_val_score(clf, X_inducing, y_inducing, cv=5)
    print(f"CV accuracy with inducing points: {scores.mean():.3f} ± {scores.std():.3f}")

    return (clf, X_test, y_test)


if __name__ == "__main__":
    main()
