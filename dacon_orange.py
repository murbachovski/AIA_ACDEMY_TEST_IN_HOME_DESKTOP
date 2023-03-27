import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, load_model, save_model, Model
from tensorflow.python.keras.layers import Dense, LSTM, Flatten, Conv1D, Conv2D, Input, Dropout, RNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
import matplotlib.pylab as plt
#1. DATA
path = ('./_data/dacon_orange/')
path_save = ('./_save/dacon_orange/')
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
#print(train_csv.shape)     # (2207, 183)
#print(test_csv.shape)      # (2208, 182)

# ISNULL
train_csv = train_csv.dropna()
#print(train_csv.shape)     # (2208, 182)

# LabelEncoder는 필요 없는 상황인 것 같고, Scaler와 원핫인코딩은 해야할까? Scaler만 해주면 좋을거 같은데? 비교해보자.
# SCALER
#scaler = MinMaxScaler()
#scaler.fit(train_csv)
#x_train = scaler.transform(train_csv)
#x_test = scaler.transform(test_csv)
#print(train_csv.shape, test_csv.shape)      
# SCALER는 데이터 분리해준 뒤에 해줘야할 거 같아. ID열이 걸리네. 아니구나 index_col=0을 안해줬구나 ㅠㅠ
# ValueError: X has 182 features, but MinMaxScaler is expecting 183 features as input. 오류가 발생하여 분리 후에 Scaler를 해보겠다.

# x, y SPLIT
x = train_csv.drop(['착과량(int)'], axis=1)
#print(x.shape)         # (2207, 182)
y = train_csv['착과량(int)']
#print(y.shape)          # (2207,)

# TRAIN_TEST_SPLIT
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    random_state=2222,
    test_size=0.3
)
print(x_train, x_test)
#print(x_train.shape, x_test.shape)      # (2151, 182) (56, 182)

# SCALER
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_train, x_test)
#print(x_train.shape, x_test.shape)     # (2151, 182) (56, 182)

#2. MODEL
model = Sequential()
model.add(Dense(256, input_shape=(182,)))
model.add(Dropout(0.6))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

#3. COMPILE
model.compile(loss = 'mae', optimizer = 'adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss',
    patience=100,
    mode='auto',
    restore_best_weights=True
)
hist = model.fit(x_train, y_train, epochs=1, batch_size=32, validation_split=0.3, callbacks=[es])
#print(y_test)
#print(y_test.shape) # (663,)

#4. PREDICT
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
if y_predict.size > 1:
    y_predict = np.argmax(y_predict, axis=1)
if y_test.size > 1:
    y_test = np.argmax(y_test, axis=1)
#print(y_test)
#print(y_test.shape)
acc = accuracy_score(y_test, y_predict)

#5. SUBMIT
test_csv = scaler.transform(test_csv)
y_submit = model.predict(test_csv)
y_submit = np.argmax(y_submit, axis=1)
submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
submission['착과량(int)'] = y_submit
submission.to_csv(path_save + 'dacon_orange_submit.csv')

# NMAE
def NMAE(y_test, y_predict):
    mae = np.mean(np.abs(y_test-y_predict))
    score = mae / np.mean(np.abs(y_test))
    return score
nmae = NMAE(y_test, y_predict)

print('loss: ', results[0], 'acc: ', results[1], 'NMAE: ', nmae)

#6. PLT
plt.plot(hist.history['acc'], label='acc', color='red')
plt.plot(hist.history['val_acc'], label='val_acc', color='blue')
plt.plot(hist.history['loss'], label='loss', color='green')
plt.plot(hist.history['val_loss'], label='val_loss', color='yellow')
plt.legend()
plt.show()