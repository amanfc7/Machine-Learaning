import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


# Decision Tree Regressor class
class DecisionTreeRegressor():
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        if (self.max_depth and depth >= self.max_depth) or len(X) <= self.min_samples_split:
            return np.mean(y)
        
        best_split = self._best_split(X, y)
        if best_split is None:
            return np.mean(y)

        left_mask = X[:, best_split['feature']] < best_split['value']
        right_mask = ~left_mask
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {'feature': best_split['feature'], 'value': best_split['value'], 'left': left_tree, 'right': right_tree}

    def _best_split(self, X, y):
        best_split = None
        best_mse = float('inf')

        for feature_index in range(X.shape[1]):
            possible_values = np.unique(X[:, feature_index])
            for value in possible_values:
                left_mask = X[:, feature_index] < value
                right_mask = ~left_mask

                left_y = y[left_mask]
                right_y = y[right_mask]

                if len(left_y) == 0 or len(right_y) == 0:
                    continue

                mse = self._mean_squared_error(left_y, right_y)
                if mse < best_mse:
                    best_mse = mse
                    best_split = {'feature': feature_index, 'value': value}

        return best_split

    def _mean_squared_error(self, left_y, right_y):
        left_mean = np.mean(left_y)
        right_mean = np.mean(right_y)
        left_mse = np.mean((left_y - left_mean) ** 2)
        right_mse = np.mean((right_y - right_mean) ** 2)
        return left_mse + right_mse

    def predict(self, X):
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)

    def _predict_single(self, x):
        node = self.tree
        while isinstance(node, dict):
            if x[node['feature']] < node['value']:
                node = node['left']
            else:
                node = node['right']
        return node


# Random Forest Regressor class
class RandomForestRegressor():
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, max_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(predictions, axis=0)


# Evaluation function
def evaluate_model(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nModel Evaluation:")
    print("Model score on test set (R²):", r2)
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("Mean Absolute Error (MAE):", mae)


# Dataset Configuration
datasets = {
    "1": {"file": "./real_estate_preprocessed.csv", "target": "Y house price of unit area"},
    "2": {"file": "./college_data_preprocessed.csv", "target": "percent_pell_grant"},
}

# Prompt the user to choose a dataset
print("Choose a dataset:")
for key, value in datasets.items():
    print(f"{key}: {value['file']} (Target: {value['target']})")

choice = input("Enter the number corresponding to your choice: ").strip()
if choice not in datasets:
    print("Invalid choice. Exiting...")
    exit()

dataset = datasets[choice]

# Load selected dataset
print(f"\nLoading dataset: {dataset['file']}...")
ds = pd.read_csv(dataset['file'])

# Data (as numpy arrays)
X = ds.drop(columns=[dataset["target"]]).to_numpy()
y = ds[dataset["target"]].to_numpy()

# Split the dataset into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the RandomForestRegressor
rf = RandomForestRegressor(n_estimators=10, max_depth=5)
rf.fit(X_train, y_train)

# Predict on the test set
y_pred = rf.predict(X_test)

# Evaluate the model
evaluate_model(y_test, y_pred)
