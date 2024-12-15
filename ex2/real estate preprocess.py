import pandas as pd
from sklearn.preprocessing import RobustScaler

# for loading the dataset:

file_path = r"C:\Users\amanf\Downloads\ML_Ex_2\Real estate valuation data set.xlsx"

df = pd.read_excel(file_path)
# df = df.drop(columns=['No', 'X1 transaction date'])      # better perfomance metrics

def preprocess_data(df):

    # to handle values in "X3 distance to the nearest MRT station" using Robust scaling:

    scaler = RobustScaler()
    
    # Now, to apply Robust scalar to "X3 distance to the nearest MRT station" column: 

    df['X3 distance to the nearest MRT station'] = scaler.fit_transform(df[['X3 distance to the nearest MRT station']])

 # to save the preprocessed dataset to a CSV file:

    preprocessed_file_path = r"C:\Users\amanf\Downloads\ML_Ex_2\csv\house_data_preprocessed.csv"
    df.to_csv(preprocessed_file_path, index=False)
    print(f"Preprocessed data saved to '{preprocessed_file_path}'.")

    return df

df = preprocess_data(df)
