import torch
import gpytorch
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, train_test_split
import numpy as np
import pandas as pd
from ds_load_util import load_dataset  # Ensure this utility is correctly implemented

# Define the Sparse GP Classification Model with multiple kernel options
class GPClassificationModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, kernel_type='RBF'):
        # Variational distribution and strategy
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        
        # Properly initialize the parent class (call to super)
        super().__init__(variational_strategy)

        # Select kernel
        if kernel_type == 'RBF':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        elif kernel_type == 'Matern':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())
        elif kernel_type == 'Linear':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")

        self.mean_module = gpytorch.means.ConstantMean()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPClassifier:
    def __init__(self, num_inducing_points=100, kernel_type='RBF'):
        self.num_inducing_points = num_inducing_points
        self.kernel_type = kernel_type
        self.model = None
        self.likelihood = gpytorch.likelihoods.BernoulliLikelihood()

    def fit(self, X_train, y_train, num_epochs=50):
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

        # Use a subset of training data as inducing points (select randomly)
        idx = torch.randperm(X_train_tensor.size(0))[:self.num_inducing_points]
        inducing_points = X_train_tensor[idx]

        # Define model and likelihood with the specified kernel
        self.model = GPClassificationModel(inducing_points, kernel_type=self.kernel_type)
        self.likelihood = gpytorch.likelihoods.BernoulliLikelihood()

        # Define optimizer and marginal log likelihood (mll)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=y_train_tensor.size(0))

        # Train the model
        self.model.train()
        self.likelihood.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            output = self.model(X_train_tensor)
            loss = -mll(output, y_train_tensor)
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item()}")

    def score(self, X_test, y_test):
        self.model.eval()
        self.likelihood.eval()

        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        with torch.no_grad():
            observed_pred = self.likelihood(self.model(X_test_tensor))
            pred_labels = observed_pred.mean > 0.5
            accuracy = (pred_labels == y_test_tensor).float().mean().item()
            return accuracy


def cross_validate(n_iter, number_samples, scaler_no, kernel_type, n_splits=5):
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

    # Load dataset
    X_train, X_test, y_train, y_test = load_dataset(
        'reviews',
        preprocess=True,
        imputer=SimpleImputer(strategy="constant", fill_value=-1),
        scaler=scaler,
    )

    # Encode y data
    y_train = preprocessing.LabelEncoder().fit_transform(y_train)
    y_test = preprocessing.LabelEncoder().fit_transform(y_test)

    # Initialize KFold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    accuracies = []

    # Cross-validation loop
    for train_idx, test_idx in kf.split(X_train):
        # Split data into training and validation for each fold
        X_train_fold, X_test_fold = X_train[train_idx], X_train[test_idx]
        y_train_fold, y_test_fold = y_train[train_idx], y_train[test_idx]

        # Initialize and train the sparse GP classifier
        classifier = GPClassifier(num_inducing_points=number_samples, kernel_type=kernel_type)
        classifier.fit(X_train_fold, y_train_fold, num_epochs=n_iter)

        # Evaluate the model
        accuracy = classifier.score(X_test_fold, y_test_fold)
        accuracies.append(accuracy)
        print(f"Fold accuracy: {accuracy}")

    # Compute and print average accuracy
    avg_accuracy = np.mean(accuracies)
    print(f"Average accuracy across {n_splits} folds: {avg_accuracy}")
    return avg_accuracy


def simple_train_test(n_iter, number_samples, scaler_no, kernel_type):
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

    # Load dataset
    X_train, X_test, y_train, y_test = load_dataset(
        'reviews',
        preprocess=True,
        imputer=SimpleImputer(strategy="constant", fill_value=-1),
        scaler=scaler,
    )

    # Encode y data
    y_train = preprocessing.LabelEncoder().fit_transform(y_train)
    y_test = preprocessing.LabelEncoder().fit_transform(y_test)

    # Split data into train-test split
    classifier = GPClassifier(num_inducing_points=number_samples, kernel_type=kernel_type)
    classifier.fit(X_train, y_train, num_epochs=n_iter)
    
    accuracy = classifier.score(X_test, y_test)
    print(f"Test accuracy: {accuracy}")
    return accuracy


def main():
    # Set hyperparameter grid
    n_iter_values = [50, 100]  # Different number of iterations (epochs)
    number_samples_values = [10000]  # Different number of inducing points
    scaler_values = [4]  # Different scalers
    kernel_types = ['RBF', 'Matern']  # Different kernel types
    
    # Flag to control cross-validation
    use_cross_validation = True  # Set to False to skip cross-validation

    best_accuracy = 0
    best_params = {}

    # Iterate over hyperparameters and perform cross-validation or simple train-test
    for n_iter in n_iter_values:
        for number_samples in number_samples_values:
            for scaler_no in scaler_values:
                for kernel_type in kernel_types:
                    print(f"\nTesting with n_iter={n_iter}, number_samples={number_samples}, scaler={scaler_no}, kernel={kernel_type}")
                    if use_cross_validation:
                        avg_accuracy = cross_validate(n_iter, number_samples, scaler_no, kernel_type)
                    else:
                        avg_accuracy = simple_train_test(n_iter, number_samples, scaler_no, kernel_type)
                    if avg_accuracy > best_accuracy:
                        best_accuracy = avg_accuracy
                        best_params = {'n_iter': n_iter, 'number_samples': number_samples, 'scaler': scaler_no, 'kernel': kernel_type}

    print("\nBest Model Configuration:")
    print(best_params)
    print(f"Best Average Accuracy: {best_accuracy}")


if __name__ == '__main__':
    main()
