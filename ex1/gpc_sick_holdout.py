import numpy as np
import pandas as pd
from itertools import product
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessClassifier as gpc
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct
from sklearn.model_selection import train_test_split
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

# Pipelines for numeric and categorical preprocessing
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
    #1.0 * DotProduct(sigma_0=.1)
]

# Define scalers
scalers = [
    StandardScaler(),
    MinMaxScaler(),
    RobustScaler()
]

# Define imputer strategies
imputer_strategies = ['mean', 'median', 'most_frequent', 'constant']

# Initialize a pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', gpc(max_iter_predict=1000000))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train-validation split from the training set
X_train_inner, X_val, y_train_inner, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

# Initialize variables to store the best results
best_model = None
best_params = None
best_val_score = 0
val_scores = []

# Iterate over all parameter combinations
for imputer_strategy, scaler, kernel in product(imputer_strategies, scalers, kernels):
    print(f"Testing combination: Imputer={imputer_strategy}, Scaler={scaler}, Kernel={kernel}")
    
    # Update numeric pipeline with current imputer strategy and scaler
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=imputer_strategy, fill_value=-1 if imputer_strategy == 'constant' else None)),
        ('scaler', scaler)
    ])
    
    # Update the main preprocessor with the modified numeric pipeline
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_columns),
        ('cat', categorical_transformer, categorical_columns)
    ])
    
    # Update the pipeline with the new preprocessor and kernel
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', gpc(kernel=kernel, max_iter_predict=1000000))
    ])
    
    # Train the pipeline on the inner training set
    pipeline.fit(X_train_inner, y_train_inner)
    
    # Evaluate on the validation set
    val_predictions = pipeline.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)
    val_scores.append(val_accuracy)
    
    print(f"Validation Accuracy: {val_accuracy:.4f}\n")
    
    # Update the best model if the current one performs better
    if val_accuracy > best_val_score:
        best_val_score = val_accuracy
        best_params = {
            'imputer_strategy': imputer_strategy,
            'scaler': scaler,
            'kernel': kernel
        }
        best_model = pipeline

# Results on the validation set
mean_val_score = np.mean(val_scores)
std_val_score = np.std(val_scores)

print("\nHoldout Validation Results:")
print(f"Best Parameters: {best_params}")
print(f"Validation Accuracy with Best Parameters: {best_val_score:.4f}")
print(f"Mean Validation Accuracy: {mean_val_score:.4f}")
print(f"Standard Deviation of Validation Accuracy: {std_val_score:.4f}")

# Retrain the best model on the entire training set
best_model.fit(X_train, y_train)

# Evaluate the best model on the test set
test_predictions = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)

print(f"\nTest Accuracy with Best Model: {test_accuracy:.4f}")
