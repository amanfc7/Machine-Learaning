import numpy as np
import pandas as pd
from itertools import product
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.gaussian_process import GaussianProcessClassifier as gpc
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data
data = pd.read_csv("/home/lea/CSE_VSC/Machine Learning/Exercise_1_classification/reviews/amazon_review_ID.shuf.lrn.csv")

# Create features and targets
X = data.drop(['ID', 'Class'], axis=1)
y = data['Class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train-validation split from the training set
X_train_inner, X_val, y_train_inner, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

# Define scalers and kernels
scalers = [
    StandardScaler(),
    MinMaxScaler(),
    RobustScaler()
]

kernels = [
    1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e10)),
    1.0 * Matern(length_scale=1.0, nu=1.5),
    1.0 * RationalQuadratic(length_scale=1.0, alpha=0.5, alpha_bounds=(1e-3, 1e6)),
    1.0 * DotProduct(sigma_0=1.0)
]

# Define imputation strategies
imputer_strategies = ['mean', 'median', 'most_frequent', 'constant']

# Pipeline
pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
    ("scaler", preprocessing.StandardScaler()),
    ('gpc', gpc(max_iter_predict=1000000))
])

# Initialize variables to store the best results
best_model = None
best_params = None
best_val_score = 0
val_scores = []

# Iterate over all parameter combinations
for imputer_strategy, scaler, kernel in product(imputer_strategies, scalers, kernels):
    # Configure imputer and scaler
    if imputer_strategy == 'constant':
        pipeline.set_params(imputer__strategy=imputer_strategy, imputer__fill_value=-1)
    else:
        pipeline.set_params(imputer__strategy=imputer_strategy)
    
    # Update pipeline parameters
    pipeline.set_params(scaler=scaler, gpc__kernel=kernel)
    
    # Train on the inner training set
    pipeline.fit(X_train_inner, y_train_inner)
    
    # Evaluate on the validation set
    val_predictions = pipeline.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)
    val_scores.append(val_accuracy)
    
    # Update the best model if current accuracy is better
    if val_accuracy > best_val_score:
        best_val_score = val_accuracy
        best_params = {
            'imputer__strategy': imputer_strategy,
            'scaler': scaler,
            'gpc__kernel': kernel
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
