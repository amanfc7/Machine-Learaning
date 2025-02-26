import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# to load and preview data:

def load_and_preview_data(file_path, is_arff=False, file_type="train"):
    print(f"Loading {file_type} data from {file_path}")
    try:
        if is_arff:
            from scipy.io import arff
            data = arff.loadarff(file_path)
            df = pd.DataFrame(data[0])
            for column in df.select_dtypes(include=['object']).columns:
                df[column] = df[column].str.decode('utf-8')
        else:
            df = pd.read_csv(file_path, header=None if file_path.endswith(".data") else 'infer', on_bad_lines='skip')

        print("\nDataset Loaded:")
        print(df.head())
        print("\nDataset Information:")
        print(df.info())
        print("\nMissing Values Count Before Handling are:")
        print(df.isin(["?", np.nan]).sum())
        return df

    except Exception as e:
        print(f"Error while loading data: {e}")
        return pd.DataFrame()

# Function to handle missing values:

def handle_missing_values(df):
    print("\nHandling Missing Values...")
    df.replace(["?"], np.nan, inplace=True)

    print("\nMissing Values Count After Replacement (Before Imputation) are:")
    missing_counts = df.isnull().sum()
    print(missing_counts)

    # to drop the columns that contain all missing values:

    all_missing_columns = missing_counts[missing_counts == len(df)].index
    if len(all_missing_columns) > 0:
        print(f"\nDropping columns with all missing values: {list(all_missing_columns)}")
        df.drop(columns=all_missing_columns, inplace=True)

    # to drop columns with all constant values (all identical or same values):

    constant_columns = [col for col in df.columns if df[col].nunique() == 1]
    if len(constant_columns) > 0:
        print(f"\nDropping columns with identical values: {list(constant_columns)}")
        df.drop(columns=constant_columns, inplace=True)

    if df.isnull().sum().sum() == 0:
        print("No missing values detected. Skipping imputation.")
        return df

    columns_with_missing = df.columns[df.isnull().any()]

    # to Impute categorical data
    categorical_columns = df.select_dtypes(include=['object']).columns.intersection(columns_with_missing)
    if len(categorical_columns) > 0:
        print(f"Imputing missing values in categorical columns: {list(categorical_columns)}")
        imputer = SimpleImputer(strategy='most_frequent')   # Using Mode
        df[categorical_columns] = imputer.fit_transform(df[categorical_columns])

    # to Impute numerical data
    numerical_columns = df.select_dtypes(include=[np.number]).columns.intersection(columns_with_missing)
    if len(numerical_columns) > 0:
        print(f"Imputing missing values in numerical columns: {list(numerical_columns)}")
        imputer = SimpleImputer(strategy='median')     # Using Median
        df[numerical_columns] = imputer.fit_transform(df[numerical_columns])

    print("\nMissing Values Count After Imputation:")
    print(df.isnull().sum())
    return df


# to encode categorical variables:

def encode_categorical(df, ordinal_columns=None):
    label_encoder = LabelEncoder()
    ordinal_encoder = OrdinalEncoder()

    for column in df.columns:
        if df[column].dtype == 'object':  # Encode categorical columns
            if ordinal_columns and column in ordinal_columns:       # Applying Ordinal Encoder if it's an ordinal column
                df[column] = ordinal_encoder.fit_transform(df[[column]])
            else:   
                df[column] = label_encoder.fit_transform(df[column])     # to apply LabelEncoder for other categorical columns
    return df

# Main preprocessing function (for preprocessing of the datasets):

def preprocess_data(df, dataset_name, ordinal_columns=None, scale_columns=None):
    print(f"\nPreprocessing Dataset: {dataset_name}")

    # to define target column and ID column based on dataset:
    if dataset_name == "Wine":
        target_column = df.columns[0]   # first column 
        id_column = None  
    elif dataset_name == "Sick":  # The 'Sick' column name in the sick dataset
        target_column = df.columns[-1]
        id_column = None  
    elif dataset_name == "CongressionalVotingID":
        target_column = "Class Name"     # The 'class' column name in the voting dataset
        id_column = df.columns[0]  # ID is the first column
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # to separate target column and ID column:

    target = df[target_column].copy()

    # to Encode the target column if it's categorical:

    print("\nEncoding Target Column...")
    if target.dtype == 'object' or isinstance(target.iloc[0], str):
        label_encoder = LabelEncoder()
        target = label_encoder.fit_transform(target)
        print(f"Target classes: {label_encoder.classes_}")

    # Drop the target column and ID column from the features:

    columns_to_drop = [target_column]
    if id_column and id_column in df.columns:
        columns_to_drop.append(id_column)
    features = df.drop(columns=columns_to_drop, errors='ignore')  # Drop target and ID columns

    # to Handle missing values:

    if features.isnull().sum().sum() > 0 or features.isin(["?"]).sum().sum() > 0:
        features = handle_missing_values(features)
    else:
        print("No missing values detected. Skipping missing value handling.")

    # to Encode categorical columns:

    features = encode_categorical(features, ordinal_columns=ordinal_columns)

    # Apply StandardScaler:
    
    if scale_columns:
   
        if dataset_name == "Wine":
        
            columns_to_scale = features.columns[1:]  
        elif dataset_name == "Sick":
        
            columns_to_scale = ['TSH', 'T3', 'TT4', 'T4U', 'FTI']
        else:
            columns_to_scale = []
    
        if len(columns_to_scale) > 0: 
            scaler = StandardScaler()
            features[columns_to_scale] = scaler.fit_transform(features[columns_to_scale])



    # to attach the ID target columns again to datasets after preprocessing:

    df_processed = features.copy()
    if id_column and id_column in df.columns:
        df_processed[id_column] = df[id_column]
    df_processed[target_column] = target 

    if dataset_name == "Wine":
        cols = [target_column] + [col for col in df_processed.columns if col != target_column]
        df_processed = df_processed[cols]

    elif dataset_name == "CongressionalVotingID":
        cols = [target_column] + [col for col in df_processed.columns if col not in [target_column]]
        df_processed = df_processed[cols]

    # to finally Return the processed DataFrame:

    return df_processed


# to Define file paths for datasets:

datasets = {
    "Wine": "./wine.data",
    "Sick": "./dataset_38_sick.arff",
    "CongressionalVotingID": "./house-votes-84.csv"
}

# Loading of datasets:

congressional_voting = load_and_preview_data(datasets["CongressionalVotingID"], file_type="train")
wine_data = load_and_preview_data(datasets["Wine"], file_type="full")
sick_data = load_and_preview_data(datasets["Sick"], is_arff=True, file_type="full")

# for Preprocessing of datasets:

wine_after = preprocess_data(wine_data.copy(), "Wine", scale_columns=True)
congressional_voting_after = preprocess_data(congressional_voting.copy(), "CongressionalVotingID")
sick_data_after = preprocess_data(sick_data.copy(), "Sick", scale_columns=True)

# to finally Save preprocessed datasets as csv files and the plots:

wine_after.to_csv("wine_preprocessed.csv", index=False)
congressional_voting_after.to_csv("congressional_voting_preprocessed.csv", index=False)
sick_data_after.to_csv("sick_data_preprocessed.csv", index=False)

save_dir = "./"
