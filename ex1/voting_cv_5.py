import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OrdinalEncoder
from sklearn.gaussian_process import GaussianProcessClassifier as gpc
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split

# 1. import data
ds = pd.read_csv('/home/lea/CSE_VSC/Machine Learning/Exercise_1_classification/congressional-voting/CongressionalVotingID.shuf.lrn.csv')
X = ds.drop(['ID', 'class'], axis=1)
y = ds['class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

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

# 2. data exploration and preprocessing
pipeline = Pipeline([
    ("encoder", OrdinalEncoder()),
    # ("encoder", preprocessing.OneHotEncoder()),
    ("imputer", SimpleImputer()),
    ("scaler", StandardScaler()),
    ('gpc', gpc(max_iter_predict=1000000))
])

# Param grid to search over
param_grid = [
    # Strategies without fill_value
    {'imputer__strategy': ['mean', 'median', 'most_frequent'], 
     'scaler': scalers, 
     'gpc__kernel': kernels},
    
    # Strategies with fill_value
    {'imputer__strategy': ['constant'], 
     'imputer__fill_value': [-1],  # Define the constant values to test
     'scaler': scalers, 
     'gpc__kernel': kernels},
]

# Set up GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)

# Perform the search
grid_search.fit(X_train, y_train)  # Make sure y_train is 1D

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Print the best parameters
print("\nBest Parameters from GridSearchCV:")
print(grid_search.best_params_)

# Extracting the mean and standard deviation of the cross-validation scores
cv_results = grid_search.cv_results_
mean_scores = cv_results['mean_test_score']  # Mean test scores from CV
std_scores = cv_results['std_test_score']    # Standard deviation of test scores

# Print the mean and standard deviation of the cross-validation scores
print("\nMean of Cross-Validation Scores for Each Parameter Combination:")
print(mean_scores)

print("\nStandard Deviation of Cross-Validation Scores for Each Parameter Combination:")
print(std_scores)

# Optionally, print the overall mean and std of all the CV results
print("\nOverall Mean CV Score:", mean_scores.mean())
print("Overall Standard Deviation of CV Scores:", std_scores.mean())

# Calculate and print the accuracy on training and test data
train_accuracy = accuracy_score(y_train, best_model.predict(X_train))  # Ensure y_train is 1D
test_accuracy = accuracy_score(y_test, best_model.predict(X_test))    # Ensure y_test is 1D

print(f"\nTraining Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")