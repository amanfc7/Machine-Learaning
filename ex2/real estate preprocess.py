import pandas as pd
from sklearn.preprocessing import RobustScaler

# for loading the dataset:

file_path = "Real_estate_valuation.csv"

df = pd.read_excel(file_path)

def preprocess_data(df):
    # to drop the "No" column:
    df = df.drop(columns=['No'], errors='ignore')  # Drop "No" column if it exists

    # to transform the "X1 transaction date" column into year and month features:

    df['Year'] = df['X1 transaction date'].astype(int)  # to extract year as integer
    df['Month'] = ((df['X1 transaction date'] % 1) * 12).round().astype(int)  # to extract month
    
    # to drop the original "X1 transaction date" column:

    df = df.drop(columns=['X1 transaction date'])  

    # to reorder columns to have "Year" and "Month" at the beginning:

    column_order = ['Year', 'Month'] + [col for col in df.columns if col not in ['Year', 'Month']]
    df = df[column_order]  

    # to handle values in "X3 distance to the nearest MRT station" using Robust scaling:

    scaler = RobustScaler()
    
    # Now, to apply Robust scalar to "X3 distance to the nearest MRT station" column: 

    df['X3 distance to the nearest MRT station'] = scaler.fit_transform(df[['X3 distance to the nearest MRT station']])

    # to save the preprocessed dataset to a CSV file:

    preprocessed_file_path = "real_estate_preprocessed.csv"
    df.to_csv(preprocessed_file_path, index=False)
    print(f"Preprocessed data saved to '{preprocessed_file_path}'.")

    return df

df = preprocess_data(df)
