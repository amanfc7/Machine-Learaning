import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
import arff

# for loading and preview the datasets:

def load(file_path, is_arff=False):
    
    print(f"Loading data from the dataset: {file_path}")
    if is_arff:
        with open(file_path, 'r') as file:
            dataset = arff.load(file)
            df = pd.DataFrame(dataset['data'], columns=[attr[0] for attr in dataset['attributes']])
    else:
        df = pd.read_csv(file_path)
    
    print("\nDataset Loaded:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
    return df

# to handle missing values for the College dataset:

def college_missing_values(df):
   
    if df.empty:
        print("\nDataset is empty. Skipping missing value handling.")
        return df
    
    print("\nHandling Missing Values for College Dataset...")
    df.replace("?", np.nan, inplace=True)

    print("\nMissing Values Count After Replacement (Before Imputation):")
    print(df.isnull().sum())

    # to drop columns (as they only contain data not useful for processing):

    if all(col in df.columns for col in ['school_webpage', 'school_name', 'zip', 'latitude', 'longitude', 'carnegie_basic_classification', 'carnegie_undergraduate', 'carnegie_size', 'religious_affiliation']):   
        df.drop(columns=['school_webpage', 'school_name', 'zip', 'latitude', 'longitude', 'carnegie_basic_classification', 'carnegie_undergraduate', 'carnegie_size', 'religious_affiliation'], inplace=True)
        print("\nDropped some non-useful columns.")


    # to drop the columns which have all missing values:

    all_missing_columns = df.columns[df.isnull().all()]
    if len(all_missing_columns) > 0:
        print(f"\nDropping columns with all missing values: {list(all_missing_columns)}")
        df.drop(columns=all_missing_columns, inplace=True)

    # to convert specific columns to numeric, coercing errors to NaN:

    columns_to_numeric = ['percent_female', 'agege24', 'faminc']
    for col in columns_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # to Impute categorical data:

    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    print(f"Imputing missing values in categorical columns: {list(categorical_columns)}")

    # to impute 'predominant_degree' column:

    if 'predominant_degree' in df.columns:
        predominant_degree_mode = df['predominant_degree'].mode()[0]
        print(f"\nImputing 'predominant_degree' column with its mode: {predominant_degree_mode}")
        df['predominant_degree'].fillna(predominant_degree_mode, inplace=True)

    imputer_categorical = SimpleImputer(strategy='most_frequent')
    df[categorical_columns] = imputer_categorical.fit_transform(df[categorical_columns])


    # to Impute numerical data:

    numerical_columns = df.select_dtypes(include=[np.number]).columns
    print(f"Imputing missing values in numerical columns: {list(numerical_columns)}")
    imputer_numerical = SimpleImputer(strategy='median')
    df[numerical_columns] = imputer_numerical.fit_transform(df[numerical_columns])

    # to ensure that there are no missing values now:

    if df.isnull().sum().any():
        print("\nWarning: There are still missing values after imputation. Re-checking...")
        print(df.isnull().sum())
    else:
        print("\nAll missing values have been successfully handled.")

    print("\nMissing Values Count After Imputation:")
    print(df.isnull().sum())
    return df

# to encode categorical variables:

def encode_categorical(df):
  
    if df.empty:
        print("\nDataset is empty. Skipping encoding.")
        return df

    print("\nEncoding Categorical Variables...")
    label_encoder = LabelEncoder()

    for column in df.select_dtypes(include=['object', 'category']).columns:
        print(f"Encoding column: {column}")
        df[column] = df[column].fillna('Unknown') 
        df[column] = label_encoder.fit_transform(df[column].astype(str))

    print("\nCategorical encoding completed.")
    return df

# Preprocessing the dataset:

def preprocess_data(df, dataset_name, handle_missing=False, target_column='percent_pell_grant', apply_variance_threshold=False):
    
    if df.empty:
        print(f"\nDataset {dataset_name} is empty. Skipping preprocessing.")
        return df

    print(f"\nPreprocessing Dataset: {dataset_name}")

    if handle_missing:
        df = college_missing_values(df)

    # Encode categorical columns:

    df = encode_categorical(df)

    # Validate target column existence:

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")
    print(f"Target column is: {target_column}")
    # to Scale relevant numerical features:

    columns_to_scale = ['undergrad_size','percent_white','percent_black', 'percent_hispanic', 'percent_part_time', 'percent_part_time_faculty', 'completion_rate', 'percent_female', 'agege24', 'faminc']  
    print(f"\nScaling numerical columns: {list(columns_to_scale)}")
    
    # Use RobustScaler:
     
    scaler = RobustScaler()
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    # to apply Feature Selection (Variance Threshold):
    if apply_variance_threshold:
        print("\nApplying Variance Threshold for feature selection...")
        selector = VarianceThreshold(threshold=0.02)  
        df = pd.DataFrame(selector.fit_transform(df), columns=df.columns[selector.get_support(indices=True)])
        print(f"Features remaining after variance threshold: {df.shape[1]}")

    # to Separate target and features:

    target = df[target_column]
    features = df.drop(columns=[target_column])

    # to Re-attach the target column to the processed features:

    df_processed = pd.concat([features, target], axis=1)
    print("\nPreprocessing complete.")
    return df_processed

# the file path for the dataset:

datasets = {
    "College": r"dataset.arff"
}

# Load datasets:

college_data = load(datasets["College"], is_arff=True)

# Preprocess datasets:

college_data_after = preprocess_data(college_data.copy(), "College", handle_missing=True, apply_variance_threshold=True)


# to save preprocessed dataset:

college_data_after.to_csv(os.path.join("college_data_preprocessed.csv"), index=False)
