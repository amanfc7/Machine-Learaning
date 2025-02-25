# Preprocessing is done by preprocess.py
# loads and provides preprocessed data sets
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def load_ds(data_set_to_load):
    match data_set_to_load:
        case 1:
            return load_ds_1()
        case 2:
            return load_ds_2()
        case 3:
            return load_ds_3()
        case 4:
            return load_ds_4()
        case default:
            return load_dummy_ds()

def load_dummy_ds():
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    train_size=0.75, test_size=0.25)
    return (X_train, y_train, X_test, y_test)

def load_ds_1():
    df = pd.read_csv('./sick_data_preprocessed.csv') 
    X = df.drop(columns=[df.columns[-1]]) 
    y = df[df.columns[-1]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)
    return (X_train, y_train, X_test, y_test)

def load_ds_2():
    df = pd.read_csv('./congressional_voting_preprocessed.csv')  
    X = df.drop(columns=[df.columns[0]])  
    y = df[df.columns[0]] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)
    return (X_train, y_train, X_test, y_test)

def load_ds_3():
    df = pd.read_csv('./dataset_60_waveform-5000.csv')  
    X = df.drop(columns=[df.columns[-1]]) 
    y = df[df.columns[-1]] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)
    return (X_train, y_train, X_test, y_test)

def load_ds_4():
    df = pd.read_csv('./wine_preprocessed.csv')  
    X = df.drop(columns=[df.columns[0]])  
    y = df[df.columns[0]]  
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)
    return (X_train, y_train, X_test, y_test)
