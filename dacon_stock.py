import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, load_model, save_model, Model
from tensorflow.python.keras.layers import Dense, LSTM, Flatten, Conv1D, Conv2D, Input, Dropout, RNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import FinanceDataReader as fdr
import os

path = './open/'
list_name = 'Stock_List.csv'
stock_list = pd.read_csv(os.path.join(path,list_name))
stock_list['종목코드'] = stock_list['종목코드'].apply(lambda x : str(x).zfill(6))


# 에라이... 이거는 참여가 불가능한 대회군요...