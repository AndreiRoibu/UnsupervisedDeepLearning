import numpy as np
import pandas as pd
from sklearn.utils import shuffle

def get_Kaggle_MNIST():
    train = pd.read_csv('../large_files/train.csv').values.astype(np.float32)
    train = shuffle(train)
    X_train = train[:-1000,1:] / 255
    y_train = train[:-1000,0].astype(np.int32) #column 0 are labels
    X_test = train[-1000:, 1:] / 255 # columns 1-785 are values between [0, 255]
    y_test = train[-1000:, 0].astype(np.int32)
    print("X train size is:", X_train.shape)
    print("y train size is:", y_train.shape)
    print("X test size is:", X_test.shape)
    print("y test size is:", y_test.shape)
    return X_train, y_train, X_test, y_test
