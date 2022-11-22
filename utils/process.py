import random
import numpy as np
import librosa
from tqdm import tqdm
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from settings import *
warnings.filterwarnings("ignore")
def extract_mfcc(file,noise):
    df = pd.read_excel(file)
    set_ = []
    label = []
    for i in tqdm(range(len(df['file']))):
        if 'train' in file:
            file_name = 'dataset/train/'+df['file'][i]
        else:
            file_name = 'dataset/test/'+df['file'][i]
        X, sample_rate = librosa.core.load(file_name)
        X = librosa.util.fix_length(X, size=242500)
        a = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=N_MFCC).T
        set_.append(a)
        label.append(df['label'][i])
        
        ## 扩充数据集，添加白噪声
        if noise and 'train' in file:
            percent = random.random()*0.5
            random_values = np.random.rand(len(X))
            src = X + percent * random_values
            a = librosa.feature.mfcc(y=src, sr=sample_rate, n_mfcc=N_MFCC).T
            set_.append(a)
            label.append(df['label'][i])
    return (set_,label)

def split_data(X, y, test_size=0.2, valid_size=0.1):
    # split training set and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=7)
    # split training set and validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid_size, random_state=7)
    # return a dictionary of values
    return {
        "X_train": X_train,
        "X_valid": X_valid,
        "X_test": X_test,
        "y_train": y_train,
        "y_valid": y_valid,
        "y_test": y_test
    }
