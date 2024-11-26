import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# target columns for each dataset:
target_columns = {
    "CongressionalVotingID": "class",  
    "Wine": "0",                       
    "AmazonReview": "Class",           
    "Sick": "Class"  
}

# Function to run Decision Tree Classification and evaluate performance:
def run_decision_tree_classification(X, y, dataset_name, use_cv=True):
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Oversample and undersample methods to balance the dataset:
    oversample = SMOTE(sampling_strategy='auto', random_state=42)
    undersample = RandomUnderSampler(sampling_strategy='auto', random_state=42)

    # Pipeline to handle oversampling and undersampling:
    pipeline = Pipeline([
        ('sampling', oversample),  
        ('undersampling', undersample),  
        ('classifier', DecisionTreeClassifier(random_state=42)) 
    ])

    # parameter grid for hyperparameter tuning:
    param_grid = {
        'classifier__criterion': ['gini', 'entropy'],  # Gini and Information Gain
        'classifier__max_depth': [None, 5, 10, 15, 20, 25, 30],
        'classifier__min_samples_split': [2, 10, 20, 30],
        'classifier__min_samples_leaf': [1, 5, 10, 15],
        'classifier__max_features': [None, 'sqrt', 'log2'],
        'classifier__class_weight': ['balanced', None]
    }

    if use_cv:
        # RandomizedSearchCV for cross-validation
        random_search = RandomizedSearchCV(
            pipeline, param_distributions=param_grid, n_iter=200, 
            cv=StratifiedKFold(n_splits=5), scoring='f1_weighted', n_jobs=-1, verbose=1, random_state=42
        )

        start_time = time.time()
        random_search.fit(X_train, y_train)
        end_time = time.time()

        # Best parameters and cross-validation score:
        best_params = random_search.best_params_
        best_model = random_search.best_estimator_
        best_score = random_search.best_score_
        print(f"\nBest Parameters (CV) for {dataset_name}: {best_params}")
        print(f"Best Cross-validation F1 Score (CV) for {dataset_name}: {best_score:.4f}")
        runtime = end_time - start_time

        # Calculate standard deviation for metrics using cross-validation:
        skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for train_idx, test_idx in skf.split(X, y):
            X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
            y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

            pipeline.fit(X_train_fold, y_train_fold)
            y_pred_fold = pipeline.predict(X_test_fold)

            accuracy_scores.append(accuracy_score(y_test_fold, y_pred_fold))
            precision_scores.append(precision_score(y_test_fold, y_pred_fold, average='weighted', zero_division=0))
            recall_scores.append(recall_score(y_test_fold, y_pred_fold, average='weighted', zero_division=0))
            f1_scores.append(f1_score(y_test_fold, y_pred_fold, average='weighted', zero_division=0))

        accuracy_std = np.std(accuracy_scores)
        precision_std = np.std(precision_scores)
        recall_std = np.std(recall_scores)
        f1_std = np.std(f1_scores)

    else:
        # Fixed parameters for holdout evaluation:
        dt_classifier = DecisionTreeClassifier(
            random_state=42, criterion='gini', max_depth=20, 
            min_samples_split=10, min_samples_leaf=5, max_features=None, 
            class_weight='balanced'
        )
        
        pipeline.set_params(classifier=dt_classifier)
        
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        end_time = time.time()

        best_model = pipeline
        test_accuracy = accuracy_score(y_test, pipeline.predict(X_test))
        print(f"Test Accuracy (Holdout) for {dataset_name}: {test_accuracy:.4f}")
        runtime = end_time - start_time

        # No standard deviation in holdout evaluation:
        accuracy_std = precision_std = recall_std = f1_std = 0.0

    # Evaluation Metrics:
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"\nBest Metrics for {dataset_name}:\n")
    print(f"Accuracy: {accuracy:.4f} ± {accuracy_std:.4f}")
    print(f"Precision: {precision:.4f} ± {precision_std:.4f}")
    print(f"Recall: {recall:.4f} ± {recall_std:.4f}")
    print(f"F1 Score: {f1:.4f} ± {f1_std:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)

    # Confusion Matrix Plot:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.7)
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f'{val}', ha='center', va='center', color='red')
    plt.title(f"Confusion Matrix for {dataset_name}")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    return best_model, runtime, accuracy, precision, recall, f1

# Function to compare holdout and cross-validation:
def compare_holdout_vs_cv(X, y, dataset_name):
    print(f"--- Holdout Evaluation for {dataset_name} ---")
    holdout_results = run_decision_tree_classification(X, y, dataset_name, use_cv=False)

    print(f"\n--- Cross-Validation Evaluation for {dataset_name} ---")
    cv_results = run_decision_tree_classification(X, y, dataset_name, use_cv=True)
    
    return holdout_results, cv_results

# Function to plot runtime and performance comparison:
def plot_metrics(results):
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    for metric in metrics:
        values = [result[metric] for result in results.values()]
        plt.figure(figsize=(10, 5))
        plt.bar(results.keys(), values, alpha=0.7, label=f'{metric.capitalize()}')
        plt.title(f"{metric.capitalize()} Comparison Across Datasets")
        plt.ylabel(metric.capitalize())
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()

# Load datasets and analyze:
datasets = {
    "AmazonReview": r"C:\Users\amanf\Downloads\ML_Ex_1\csv\amazon_review_preprocessed.csv",
    "Wine": r"C:\Users\amanf\Downloads\ML_Ex_1\csv\wine_preprocessed.csv",
    "Sick": r"C:\Users\amanf\Downloads\ML_Ex_1\csv\sick_data_preprocessed.csv",
    "CongressionalVotingID": r"C:\Users\amanf\Downloads\ML_Ex_1\csv\congressional_voting_preprocessed.csv"
}

results = {}
for dataset_name, file_path in datasets.items():
    try:
        df_processed = pd.read_csv(file_path)
        target_column = target_columns[dataset_name]
        X = df_processed.drop(columns=[target_column])
        y = df_processed[target_column]
        print(f"\nEvaluating {dataset_name}...")
        holdout_results, cv_results = compare_holdout_vs_cv(X, y, dataset_name)
        results[dataset_name] = {
            'holdout_runtime': holdout_results[1],
            'cv_runtime': cv_results[1],
            'accuracy': cv_results[2],
            'precision': cv_results[3],
            'recall': cv_results[4],
            'f1': cv_results[5],
        }
    except Exception as e:
        print(f"Error processing {dataset_name}: {e}")

# Plot the metrics comparison:
plot_metrics(results)
