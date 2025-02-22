
# TODO: loads and provides data sets, does preprocessing
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
    return load_dummy_ds() #TODO: replace with proper code for loading data set 1 

def load_ds_2():
    return load_dummy_ds() #TODO: replace with proper code for loading data set 2

def load_ds_3():
    return load_dummy_ds() #TODO: replace with proper code for loading data set 3 

def load_ds_4():
    return load_dummy_ds() #TODO: replace with proper code for loading data set 4 