
# 삼성전자와 현대자동차 주가로 삼성전자 주가 맞추기

# 각각 데이터에서 컬럼 7개 이상 추출(그 중 거래량은 반드시 들어갈 것)
# timesteps와 feature는 알아서 잘라라

# 제공된 데이터 외 추가 데이터 사용금지

# 1. 삼성전자 28일(화) 종가 맞추기 (점수 배점 0.3)
# 2. 삼성전자 29일(수) 아침 시가 맞추기 (점수 배점 0.7)

import numpy as np
from sklearn import datasets
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Input, Dense, LSTM, Conv1D, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import time
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import datetime
import matplotlib.pyplot as plt

date = datetime.datetime.now()
date = date.strftime('%m%d_%M%M')
filepath = ('./_save/samsung/')
filename = '{val_loss:.4f}.hdf5'

#1. DATA
path = './_data/시험/'

dataset_sam = pd.read_csv(path + '삼성전자.csv', thousands=',', encoding='UTF8')
dataset_hyun = pd.read_csv(path + '현대자동차.csv', thousands=',', encoding='UTF8')

dataset_sam = dataset_sam.drop(['전일비','금액(백만)','신용비','외인(수량)','프로그램','외인비'], axis=1)
dataset_hyun = dataset_hyun.drop(['전일비','금액(백만)','신용비','외인(수량)','프로그램','외인비'], axis=1)

# dataset_sam.info()
# dataset_hyun.info()
dataset_sam = dataset_sam.fillna(0)
dataset_hyun = dataset_hyun.fillna(0)

dataset_sam = dataset_sam.loc[dataset_sam['일자'] >= '2018/05/04'] # 액면분할 이후 데이터만 사용
dataset_hyun = dataset_hyun.loc[dataset_hyun['일자'] >= '2018/05/04'] #삼성의 액면분할 날짜 이후의 행개수에 맞춰줌
#print(dataset_sam.shape, dataset_hyun.shape) # (1206, 11) (1206, 11)

dataset_sam = dataset_sam.sort_values(by=['일자'], axis=0, ascending=True) # 오름차순 정렬
dataset_hyun = dataset_hyun.sort_values(by=['일자'], axis=0, ascending=True)
#print(dataset_hyun.head) # 앞 다섯개만 보기

feature_cols = ['시가', '고가', '저가', '기관', '거래량', '외국계', '종가']

dataset_sam = dataset_sam[feature_cols]
dataset_hyun = dataset_hyun[feature_cols]
dataset_sam = np.array(dataset_sam)
dataset_hyun = np.array(dataset_hyun)

def split_xy(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column - 1

        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, : -1]
        tmp_y = dataset[x_end_number-1: y_end_number, -1]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

SIZE = 3
COLSIZE = 3
x1, y1 = split_xy(dataset_sam, SIZE, COLSIZE)
x2, y2 = split_xy(dataset_hyun, SIZE, COLSIZE)
#print(x1.shape, y1.shape) # (1202, 3, 6) (1202, 3)

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
    x1,
    x2,
    y1,
    test_size=0.2,
    shuffle=False
)

#DATA SCALER
scaler = MinMaxScaler()
# print(x1_train.shape, x1_test.shape)
# print(x2_train.shape, x2_test.shape)
# print(y_train.shape, y_test.shape)
# (961, 3, 6) (241, 3, 6)
# (961, 3, 6) (241, 3, 6)
# (961, 3) (241, 3)
x1_train = x1_train.reshape(961*3, 6)
x1_train = scaler.fit_transform(x1_train)
x1_test = x1_test.reshape(241*3, 6)
x1_test = scaler.transform(x1_test)

x2_train = x2_train.reshape(961*3, 6)
x2_train = scaler.fit_transform(x2_train)
x2_test = x2_test.reshape(241*3, 6)
x2_test = scaler.transform(x2_test)

#Conv1D에 넣기 위해 3차원화
x1_train = x1_train.reshape(961, 3, 6)
x1_test = x1_test.reshape(241, 3, 6)
x2_train = x2_train.reshape(961, 3, 6)
x2_test = x2_test.reshape(241, 3, 6)

#2. MODEL
input1 = Input(shape=(3, 6))
conv1 = Conv1D(128, 2, activation='relu')(input1)
lstm1 = LSTM(128, activation='relu')(conv1)
dense1 = Dense(128, activation='relu')(lstm1)
drop4 = Dropout(0.3)(dense1)
dense2 = Dense(128, activation='relu')(drop4)
drop5 = Dropout(0.3)(dense2)
dense3 = Dense(64, activation='relu')(drop5)
output1 = Dense(64, activation='relu')(dense3)

#2-2. MODEL2
input2 = Input(shape=(3, 6))
conv2 = Conv1D(128, 2, activation='relu')(input2)
lstm2 = LSTM(128, activation='relu')(conv2)
drop1 = Dropout(0.3)(lstm2)
dense4 = Dense(128, activation='relu')(drop1)
drop2 = Dropout(0.3)(dense4)
dense5 = Dense(128, activation='relu')(drop2)
drop3 = Dropout(0.3)(dense5)
output2 = Dense(64, activation='relu')(drop3)

from tensorflow.python.keras.layers import concatenate
merge1 = concatenate([output1, output2])
merge2 = Dense(128)(merge1)
merge3 = Dense(64, name='mg3')(merge2)
last_output = Dense(1)(merge3)

model = Model(inputs=[input1, input2], outputs=[last_output])
model.summary()

#3. COMPILE
model.compile(loss = 'mse', optimizer='adam')
#tb_hist = TensorBoard(log_dir='./_save/samsung/', histogram_freq=0, write_graph=True, write_images=True)
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath="".join([filepath, 'samsung', date, '_', filename])
)
start_time = time.time()
Es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=50,
    restore_best_weights=True
)
hist = model.fit([x1_train, x2_train], y_train, epochs=100, batch_size=64, callbacks=[Es], validation_split=0.1)
end_time = time.time()
model.save('./_save/samsung/keras53_samsung2_kdj2.h5')

#4. PREDICT
loss = model.evaluate([x1_test, x2_test], y_test)
predict = model.predict([x1_test, x2_test])
print('loss: ', loss)
print('predict: ', predict[-1:])
print('걸린 시간: ', end_time - start_time)

#6. PLT
plt.plot(hist.history['loss'], label='loss', color='red')
plt.plot(hist.history['val_loss'], label='val_loss', color='blue')
plt.legend()
plt.show()

# 1. predict:  [[61270.86]] epochs=100, batch=64
