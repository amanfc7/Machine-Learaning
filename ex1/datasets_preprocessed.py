import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

# Function to load and preview data
def load_and_preview_data(file_path, is_arff=False, file_type="train"):
    print(f"Loading {file_type} data from {file_path}")

    if is_arff:
        # Load ARFF file
        from scipy.io import arff
        data = arff.loadarff(file_path)
        df = pd.DataFrame(data[0])
        # Decode byte columns to string if necessary
        for column in df.select_dtypes(include=['object']).columns:
            df[column] = df[column].str.decode('utf-8')
    else:
        # Load CSV file
        df = pd.read_csv(file_path, header=None if file_path.endswith(".data") else 'infer')

    print("\nDataset Loaded:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
    print("\nMissing Values Count Before Handling:")
    print(df.isin(["?", "unknown", "None", np.nan]).sum())  
    return df

# Function to handle missing values
def handle_missing_values(df):
    print("\nHandling Missing Values...")
    # Replace "?", "unknown", and "None" with NaN
    df.replace(["?", "unknown", "None"], np.nan, inplace=True)

    print("\nMissing Values Count After Replacement (Before Imputation):")
    missing_counts = df.isnull().sum()
    print(missing_counts)

    # Skip handling if there are no missing values
    if missing_counts.sum() == 0:
        print("No missing values detected. Skipping imputation.")
        return df

    # Handle only columns with missing values
    columns_with_missing = missing_counts[missing_counts > 0].index

    # Impute categorical data (most frequent)
    categorical_columns = df.select_dtypes(include=['object']).columns.intersection(columns_with_missing)
    if len(categorical_columns) > 0:
        print(f"Imputing missing values in categorical columns: {list(categorical_columns)}")
        imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_columns] = imputer.fit_transform(df[categorical_columns])

    # Impute numerical data (median)
    numerical_columns = df.select_dtypes(include=[np.number]).columns.intersection(columns_with_missing)
    if len(numerical_columns) > 0:
        print(f"Imputing missing values in numerical columns: {list(numerical_columns)}")
        imputer = SimpleImputer(strategy='median')
        df[numerical_columns] = imputer.fit_transform(df[numerical_columns])

    print("\nMissing Values Count After Imputation:")
    print(df.isnull().sum())
    return df

# Function for encoding categorical variables
def encode_categorical(df):
    label_encoder = LabelEncoder()
    for column in df.columns:
        if df[column].dtype == 'object':  # Encode categorical columns
            df[column] = label_encoder.fit_transform(df[column])
    return df

# Function for scaling numerical columns
def scale_numerical(df):
    scaler = MinMaxScaler()
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df

# Preprocessing function for training datasets
def preprocess_data(df, dataset_name):
    print(f"\nPreprocessing Dataset: {dataset_name}")

    # Identify the target column based on the dataset
    if dataset_name == "AmazonReview":
        target_column = df.columns[-1]  # Last column (Class)
    elif dataset_name == "Wine":
        target_column = df.columns[0]  # First column
    elif dataset_name == "Tracks":
        target_column = "other_class"  # Column named 'other_class'
    elif dataset_name == "CongressionalVotingID":
        target_column = "class"  # Column named 'class'
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Separate target and features
    target = df[target_column].copy()
    features = df.drop(columns=[target_column])

    # Handle missing values
    if features.isnull().sum().sum() > 0 or features.isin(["?", "unknown", "None"]).sum().sum() > 0:
        features = handle_missing_values(features)
    else:
        print("No missing values detected. Skipping missing value handling.")

    # Encode categorical columns
    features = encode_categorical(features)

    # Scale numerical columns (only for certain datasets like 'Wine')
    if dataset_name == "Wine":
        features = scale_numerical(features)

    # Combine features and target into a single dataframe
    df = features.copy()
    df[target_column] = target
    return df

# Minimal preprocessing for test datasets
def preprocess_test_data(df):
    print("\nMinimal Preprocessing for Test Data")
    # Only handle missing values
    df = handle_missing_values(df)
    return df

# Function to plot comparison before and after preprocessing
def plot_comparison(before, after, dataset_name):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    axs[0].set_title(f'{dataset_name} - Before Preprocessing')
    axs[1].set_title(f'{dataset_name} - After Preprocessing')

    # Select numeric data
    before_numeric = before.select_dtypes(include=[np.number]).iloc[:, :5]
    after_numeric = after.select_dtypes(include=[np.number]).iloc[:, :5]

    # Convert to numeric if necessary
    before_numeric = before_numeric.apply(pd.to_numeric, errors='coerce')
    after_numeric = after_numeric.apply(pd.to_numeric, errors='coerce')

    # Plot mean of numeric features
    before_numeric.mean().plot(kind='bar', ax=axs[0], alpha=0.7, color='skyblue')
    after_numeric.mean().plot(kind='bar', ax=axs[1], alpha=0.7, color='orange')

    axs[0].set_xlabel('Features')
    axs[0].set_ylabel('Mean Value')
    axs[1].set_xlabel('Features')
    axs[1].set_ylabel('Mean Value')

    plt.tight_layout()
    plt.show()

# Define file paths for datasets
datasets = {
    "CongressionalVotingID_train": r"C:\Users\amanf\Downloads\ML_Ex_1\CongressionalVotingID.shuf.lrn.csv",
    "CongressionalVotingID_test": r"C:\Users\amanf\Downloads\ML_Ex_1\CongressionalVotingID.shuf.tes.csv",
    "Wine": r"C:\Users\amanf\Downloads\ML_Ex_1\wine.data",
    "AmazonReview_train": r"C:\Users\amanf\Downloads\ML_Ex_1\amazon_review_ID.shuf.lrn.csv",
    "AmazonReview_test": r"C:\Users\amanf\Downloads\ML_Ex_1\amazon_review_ID.shuf.tes.csv",
    "Tracks": r"C:\Users\amanf\Downloads\ML_Ex_1\tracks.csv",
}

# Load datasets
congressional_train = load_and_preview_data(datasets["CongressionalVotingID_train"], file_type="train")
wine_data = load_and_preview_data(datasets["Wine"], file_type="full")
amazon_train = load_and_preview_data(datasets["AmazonReview_train"], file_type="train")
tracks_data = load_and_preview_data(datasets["Tracks"], file_type="full")

# Preprocess train datasets
congressional_train_after = preprocess_data(congressional_train.copy(), "CongressionalVotingID")
wine_data_after = preprocess_data(wine_data.copy(), "Wine")
amazon_train_after = preprocess_data(amazon_train.copy(), "AmazonReview")
tracks_data_after = preprocess_data(tracks_data.copy(), "Tracks")

# Preprocess test datasets
congressional_test_after = preprocess_test_data(load_and_preview_data(datasets["CongressionalVotingID_test"], file_type="test"))
amazon_test_after = preprocess_test_data(load_and_preview_data(datasets["AmazonReview_test"], file_type="test"))

# Generate plots for train datasets
plot_comparison(congressional_train, congressional_train_after, "Congressional Voting Train")
plot_comparison(wine_data, wine_data_after, "Wine")
plot_comparison(amazon_train, amazon_train_after, "Amazon Review Train")
plot_comparison(tracks_data, tracks_data_after, "Tracks")

# Save the preprocessed data to CSV files for later use
congressional_train_after.to_csv(r"C:\Users\amanf\Downloads\ML_Ex_1\csv\congressional_train_after.csv", index=False)
wine_data_after.to_csv(r"C:\Users\amanf\Downloads\ML_Ex_1\csv\ wine_data_after.csv", index=False)
amazon_train_after.to_csv(r"C:\Users\amanf\Downloads\ML_Ex_1\csv\amazon_train_after.csv", index=False)
tracks_data_after.to_csv(r"C:\Users\amanf\Downloads\ML_Ex_1\csv\tracks_data_after.csv", index=False)

congressional_test_after.to_csv(r"C:\Users\amanf\Downloads\ML_Ex_1\csv\congressional_test_after.csv", index=False)
amazon_test_after.to_csv(r"C:\Users\amanf\Downloads\ML_Ex_1\csv\amazon_test_after.csv", index=False)
