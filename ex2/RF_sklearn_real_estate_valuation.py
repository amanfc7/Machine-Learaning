import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Dataset Configuration
df = pd.read_csv("./real_estate_preprocessed.csv")

X = df.drop(columns=['Y house price of unit area']).to_numpy()
y = df['Y house price of unit area'].to_numpy()

# Split the dataset into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


def train_model(X_train, X_test, y_train, y_test, do_gridsearch=False, apply_pca=False, n_components=None):
    # Apply PCA if enabled
    if apply_pca:
        pca = PCA(n_components=n_components)
        print(f"Applying PCA with n_components={n_components}...")
            
        # Fit PCA on the training set and transform both sets
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

        print(f"Explained Variance Ratio (first {n_components} components): {pca.explained_variance_ratio_}")
        print(f"Total Explained Variance: {sum(pca.explained_variance_ratio_):.2f}")



    if do_gridsearch:
        # GridSearch
        parameters = {
            "n_estimators" : [1, 10, 100, 1000],
            "max_features" : ['auto', 'sqrt', 'log2'],
            "max_depth" : ['None', 10, 50, 100],
            "min_samples_split" : [2, 10, 20],
            "min_samples_leaf" : [1, 2, 4],
            "max_leaf_nodes" : [None, 10, 100]
        }

        grid_search = GridSearchCV(
            RandomForestRegressor(),
            parameters,
            # n_iter=40
            cv = 5, 
            n_jobs=-1, 
            verbose=1
        )

  
        print("Performing grid search...")
        t0 = time.time()
        grid_search.fit(X_train, y_train)
        print(f"Grid search completed in {time.time() - t0:.5f} seconds")
        print("Best parameters:", grid_search.best_params_)

        clf = grid_search.best_estimator_

        
    else:
        clf = RandomForestRegressor()
        print("Training RandomForest without GridSearch...")
        t0 = time.time()
        clf.fit(X_train, y_train)
        print(f"Training completed in {time.time() - t0:.5f} seconds")

    # Predictions
    t0 = time.time()
    y_pred = clf.predict(X_test)
    print(f"Prediction completed in {time.time() - t0:.3f} seconds")

    # Evaluation
    evaluate_model(y_test, y_pred)
       

# Evaluation function
def evaluate_model(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nModel Evaluation:")
    print("Model score on test set (R²):", r2)
    print("Mean Squared Error (MSE):", mse)
    print("Mean Absolute Error (MAE):", mae)


train_model(X_train, X_test, y_train, y_test, do_gridsearch=True, apply_pca=False, n_components=25)
