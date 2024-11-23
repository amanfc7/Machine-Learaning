import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessClassifier as gpc
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml


# Fetch the dataset from OpenML
data = fetch_openml(data_id=38, as_frame=True)
X = data.data  # Features
y = data.target  # Target variable

# Replace '?' with NaN for missing values
X = X.replace('?', np.nan)

# Drop non-predictive columns
columns_to_drop = ['TBG_measured', 'TBG']
X = X.drop(columns=columns_to_drop)

# Separate categorical and numeric columns
categorical_columns = X.select_dtypes(include=['object']).columns
numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns

# Pipeline for numeric and categorical preprocessing
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Replace missing with mean
    ('scaler', StandardScaler())                 # Standardize numeric values
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Replace missing with most frequent
    ('encoder', OneHotEncoder(handle_unknown='ignore'))    # Encode categorical values
])

# Combine transformations
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_columns),
    ('cat', categorical_transformer, categorical_columns)
])

# Define kernels for Gaussian Process Classifier
kernels = [
    1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e10)),
    1.0 * Matern(length_scale=1.0, nu=1.5),
    1.0 * RationalQuadratic(length_scale=1.0, alpha=0.5, alpha_bounds=(1e-3, 1e6)),
    1.0 * DotProduct(sigma_0=1.0)
]

# Full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', gpc(max_iter_predict=1000000))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Define hyperparameter grid
param_grid = {
    'classifier__kernel': kernels
}

# Set up GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=2)

# Perform the search
grid_search.fit(X_train, y_train)

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