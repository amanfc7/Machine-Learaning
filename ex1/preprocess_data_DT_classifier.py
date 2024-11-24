import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold

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
        print(df.isin(["?", "unknown", np.nan]).sum())
        return df

    except Exception as e:
        print(f"Error while loading data: {e}")
        return pd.DataFrame()

# Function to handle missing values:

def handle_missing_values(df):
    print("\nHandling Missing Values...")
    df.replace(["?", "unknown"], np.nan, inplace=True)

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

def preprocess_data(df, dataset_name, ordinal_columns=None):
    print(f"\nPreprocessing Dataset: {dataset_name}")

    # to define target column and ID column based on dataset:
    if dataset_name == "AmazonReview":
        target_column = df.columns[-1]  # last column
        id_column = None  
    elif dataset_name == "Wine":
        target_column = df.columns[0]   # first column 
        id_column = None  
    elif dataset_name == "Sick":  # The 'Sick' column name in the sick dataset
        target_column = df.columns[-1]
        id_column = None  
    elif dataset_name == "CongressionalVotingID":
        target_column = "class"     # The 'class' column name in the voting dataset
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

    if features.isnull().sum().sum() > 0 or features.isin(["?", "unknown"]).sum().sum() > 0:
        features = handle_missing_values(features)
    else:
        print("No missing values detected. Skipping missing value handling.")

    # to Encode categorical columns:

    features = encode_categorical(features, ordinal_columns=ordinal_columns)

    # to Check the variance of features before applying VarianceThreshold:

    print("\nVariance of features before applying VarianceThreshold:")
    print(features.var().head())  

    # to apply VarianceThreshold for Amazon Review dataset (for optimizing its feature size):

    if dataset_name == "AmazonReview":
        print("\nApplying Feature Selection for Amazon Review dataset...")
        
        original_columns = features.columns
        
        # for Removing features with low variance:

        variance_threshold = VarianceThreshold(threshold=0.02)  
        features_transformed = variance_threshold.fit_transform(features)
        
        features = pd.DataFrame(features_transformed, columns=original_columns[variance_threshold.get_support()])
        
        # to Check how many features were retained:

        print(f"Number of features after low-variance filtering: {features.shape[1]}")
        
    else:
        print("\nSkipping Feature Selection for this dataset.")

    # to attach the ID target columns again to datasets after preprocessing:

    df_processed = features.copy()
    if id_column and id_column in df.columns:
        df_processed[id_column] = df[id_column]
    df_processed[target_column] = target 
  
    if dataset_name == "Wine":
        cols = [target_column] + [col for col in df_processed.columns if col != target_column]
        df_processed = df_processed[cols]

    elif dataset_name == "CongressionalVotingID":
        cols = [id_column] + [target_column] + [col for col in df_processed.columns if col not in [id_column, target_column]]
        df_processed = df_processed[cols]

    # to finally Return the processed DataFrame:

    return df_processed

# Function to plot comparison for missing values, encoding, and other aspects:

def plot_comparison(before, after, dataset_name):
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))

    # 1st Plot: Missing Values Before and After Preprocessing:
    missing_before = before.isin(["?", "unknown", np.nan]).sum()
    missing_after = after.isin(["?", "unknown", np.nan]).sum()

    missing_before = missing_before[missing_before > 0]
    missing_after = missing_after[missing_after > 0]

    axs[0, 0].bar(missing_before.index, missing_before.values, alpha=0.7, color='skyblue', label='Before')
    axs[0, 0].bar(missing_after.index, missing_after.values, alpha=0.7, color='orange', label='After')
    axs[0, 0].set_title(f'{dataset_name} - Missing Values Before and After Preprocessing')
    axs[0, 0].set_xlabel('Features')
    axs[0, 0].set_ylabel('Missing Values Count')
    axs[0, 0].legend()

    # 2nd Plot: Number of Categorical and Numerical Columns Before and After Encoding:
    categorical_before = before.select_dtypes(include=['object']).shape[1]
    numerical_before = before.select_dtypes(include=[np.number]).shape[1]
    categorical_after = after.select_dtypes(include=['object']).shape[1]
    numerical_after = after.select_dtypes(include=[np.number]).shape[1]

    axs[0, 1].bar(['Categorical', 'Numerical'], [categorical_before, numerical_before], alpha=0.7, color='skyblue', label='Before')
    axs[0, 1].bar(['Categorical', 'Numerical'], [categorical_after, numerical_after], alpha=0.7, color='orange', label='After')
    axs[0, 1].set_title(f'{dataset_name} - Categorical vs. Numerical Columns Before and After Encoding')
    axs[0, 1].set_ylabel('Count of Columns')
    axs[0, 1].legend()

    # 3rd Plot: Distribution of Target Column Before and After Encoding:
    if 'target_column' in before.columns:
        axs[1, 0].set_title(f'{dataset_name} - Target Column Distribution Before and After Encoding')
        before_target_dist = before['target_column'].value_counts()
        after_target_dist = after['target_column'].value_counts()
        before_target_dist.plot(kind='bar', ax=axs[1, 0], alpha=0.7, color='skyblue', label='Before')
        after_target_dist.plot(kind='bar', ax=axs[1, 0], alpha=0.7, color='orange', label='After')
        axs[1, 0].set_ylabel('Frequency')
        axs[1, 0].legend()

    # 4th Plot: Feature Count Before and After Preprocessing:
    feature_before_count = before.shape[1] - 1  
    feature_after_count = after.shape[1] - 1  
    axs[1, 1].bar(['Before Preprocessing', 'After Preprocessing'], [feature_before_count, feature_after_count], alpha=0.7, color='skyblue')
    axs[1, 1].set_title(f'{dataset_name} - Number of Features Before and After Preprocessing')
    axs[1, 1].set_ylabel('Number of Features')

    plt.tight_layout()
    plt.show()



# to Define file paths for datasets:

datasets = {
    "CongressionalVotingID": r"C:\Users\amanf\Downloads\ML_Ex_1\CongressionalVotingID.shuf.lrn.csv",
    "Wine": r"C:\Users\amanf\Downloads\ML_Ex_1\wine.data",
    "AmazonReview": r"C:\Users\amanf\Downloads\ML_Ex_1\amazon_review_ID.shuf.lrn.csv",
    "Sick": r"C:\Users\amanf\Downloads\ML_Ex_1\dataset_38_sick.arff",
}

# Loading of datasets:

congressional_voting = load_and_preview_data(datasets["CongressionalVotingID"], file_type="train")
wine_data = load_and_preview_data(datasets["Wine"], file_type="full")
amazon = load_and_preview_data(datasets["AmazonReview"], file_type="train")
sick_data = load_and_preview_data(datasets["Sick"], is_arff=True, file_type="full")

# for Preprocessing of datasets:

amazon_after = preprocess_data(amazon.copy(), "AmazonReview")
wine_after = preprocess_data(wine_data.copy(), "Wine")
congressional_voting_after = preprocess_data(congressional_voting.copy(), "CongressionalVotingID")
sick_data_after = preprocess_data(sick_data.copy(), "Sick")

# to finally Save preprocessed datasets as csv files: 

amazon_after.to_csv(r"C:\Users\amanf\Downloads\ML_Ex_1\csv\amazon_review_preprocessed.csv", index=False)
wine_after.to_csv(r"C:\Users\amanf\Downloads\ML_Ex_1\csv\wine_preprocessed.csv", index=False)
congressional_voting_after.to_csv(r"C:\Users\amanf\Downloads\ML_Ex_1\csv\congressional_voting_preprocessed.csv", index=False)
sick_data_after.to_csv(r"C:\Users\amanf\Downloads\ML_Ex_1\csv\sick_data_preprocessed.csv", index=False)

# to Plot the comparisons:

plot_comparison(amazon, amazon_after, "Amazon Review")
plot_comparison(wine_data, wine_after, "Wine")
plot_comparison(congressional_voting, congressional_voting_after, "Congressional Voting ID")
plot_comparison(sick_data, sick_data_after, "Sick")
