import numpy as np
import pandas as pd
import seaborn as sns
from itertools import product
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OrdinalEncoder
from sklearn.gaussian_process import GaussianProcessClassifier as gpc
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split


# Import data
ds = pd.read_csv('/home/lea/CSE_VSC/Machine Learning/Exercise_1_classification/congressional-voting/CongressionalVotingID.shuf.lrn.csv')
X = ds.drop(['ID', 'class'], axis=1)
y = ds['class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train-validation split from the training set
X_train_inner, X_val, y_train_inner, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

# Define scalers and kernels
scalers = [
    StandardScaler(),
    MinMaxScaler(),
    RobustScaler(),
]

kernels = [
    1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e10)),
    1.0 * Matern(length_scale=1.0, nu=1.5),
    1.0 * RationalQuadratic(length_scale=1.0, alpha=1.0, alpha_bounds=(1e-3, 1e10)),
    1.0 * DotProduct(sigma_0=1.0)
]

# Create the pipeline
pipeline = Pipeline([
    ("encoder", OrdinalEncoder()),
    # ("encoder", preprocessing.OneHotEncoder()),
    ("imputer", SimpleImputer()),
    ("scaler", StandardScaler()),
    ('gpc', gpc(max_iter_predict=1000000))
])

# Initialize variables to store the best results
best_params = None
best_val_score = 0
best_model = None
val_scores = []  # To store all validation scores

# Iterate over all parameter combinations
for scaler, kernel in product(scalers, kernels):
    # Update the pipeline with the current parameter combination
    pipeline.set_params(scaler=scaler, gpc__kernel=kernel)
    
    # Fit the pipeline on the inner training set
    pipeline.fit(X_train_inner, y_train_inner)
    
    # Evaluate on the validation set
    val_predictions = pipeline.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)
    
    # Append the validation accuracy to the list
    val_scores.append(val_accuracy)
    
    # Update the best model if the current one is better
    if val_accuracy > best_val_score:
        best_val_score = val_accuracy
        best_params = {'scaler': scaler, 'gpc__kernel': kernel}
        best_model = pipeline

# Calculate the mean and standard deviation of validation scores
mean_val_score = np.mean(val_scores)
std_val_score = np.std(val_scores)

# Print the best parameters, mean, and standard deviation
print("\nBest Parameters from Holdout Validation:")
print(best_params)
print(f"Validation Accuracy with Best Parameters: {best_val_score:.4f}")
print(f"Mean Validation Accuracy: {mean_val_score:.4f}")
print(f"Standard Deviation of Validation Accuracy: {std_val_score:.4f}")

# Retrain the best model on the entire training set
best_model.fit(X_train, y_train)

# Evaluate the retrained model on the test set
test_predictions = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)

print(f"\nTest Accuracy with Best Model: {test_accuracy:.4f}")