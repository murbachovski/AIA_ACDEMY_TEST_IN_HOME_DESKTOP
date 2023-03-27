import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, load_model, save_model, Model
from tensorflow.python.keras.layers import Dense, LSTM, Flatten, Conv1D, Conv2D, Input, Dropout, RNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. DATA
path = ('./_data/uos/')
path_save = ('./_save/uos/')

train_csv = pd.read_csv(path + 'train.csv')
print(train_csv)
print(train_csv.shape) # (1461, 5)

# 공공데이터를 가져와서 만들어야하는 대회였다..