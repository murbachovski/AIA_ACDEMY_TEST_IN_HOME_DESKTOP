#test GITHUB AND 복습
#23_03_14 정삭적으로 커밋이 됩니다. 윈도우 좋네...
#정리한 내용들 살펴봅시다.
# <과적합 배제>
# 데이터를 많이 넣는다.
# 노드의 일부를 뺀다. dropout
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, Model, Input, load_model
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from keras.utils import to_categorical #ONE_HOT_ENCODING
from sklearn.preprocessing import RobustScaler #SCALER
import matplotlib.pyplot as plt
import datetime #시간으로 저장해주는 고마운 녀석
from sklearn.ensemble import RandomForestClassifier


date = datetime.datetime.now()
# print(date) #2023-03-14 22:02:28.099457
date = date.strftime('%m%d_%M%M')
filepath = ('./_save/MCP/call/')
filename = '{epoch:04d}_{val_loss:.4f}_{val_acc:.4f}.hdf5'

#1. DATA
path = ('./_data/call/')
path_save = ('./_save/call/')

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

# ISNULL
train_csv = train_csv.dropna()

# x, y SPLIT
x = train_csv.drop(['전화해지여부'], axis=1)
y = train_csv['전화해지여부']

#rf = RandomForestClassifier(class_weight='balanced')
#rf.fit(y, y)
#print(np.unique(y, return_counts=True))
#classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
#classifier.fit(x, y)
#print(np.unique(x, return_counts=True))

# ONE_HOT_ENCODING
y = to_categorical(y)

# TRAIN_TEST_SPLIT()
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.25,
    random_state=4444,
    stratify=y
)
#classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
#classifier.fit(x_train, y_train)

# SCALER
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. MODEL
model = Sequential()
model.add(Dense(64, input_shape=(12,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(2, activation='softmax'))

#3. COMPILE
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=200,
    restore_best_weights=True
)
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath="".join([filepath, 'call', date, '_', filename])
)
hist = model.fit(x_train, y_train, epochs=1000, validation_split=0.25, batch_size=200, callbacks=[es])

# SAVE_MODEL
model.save('./_save/call/call2_model.h5')
# LOAD_MODEL
#model = load_model('_save\MCP\dacon_wine\dacon_wine0315_1010_0213_0.9815_0.6136.hdf5')
'''
es = EarlyStopping(
    monitor='val_acc',
    mode='auto',
    patience=500,
    restore_best_weights=True
)
mcp = ModelCheckpoint(
    monitor='val_acc',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath="".join([filepath, 'dacon_wine', date, '_', filename])
)
hist = model.fit(x_train, y_train, epochs=2000, validation_split=0.3, batch_size=200, callbacks=[es, mcp])
'''

#4. PREDICT
results = model.evaluate(x_test, y_test)
print('loss: ', results[0], 'acc:', results[1])
y_predict = model.predict(x_test)
#y_predict = classifier.predict(x_test)
y_predict_acc = np.argmax(y_predict, axis=1)
y_test_acc = np.argmax(y_test, axis=1)
#cm = confusion_matrix(y_test, y_predict)
acc = accuracy_score(y_test_acc, y_predict_acc)
f1 = f1_score(y_test_acc, y_predict_acc)
print('ACC: ', acc, 'f1: ', f1)

#5. SUBMIT
test_csv_sc = scaler.transform(test_csv)
y_submit = model.predict(test_csv_sc)
y_submit = np.argmax(y_submit, axis=1)
submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
submission['전화해지여부'] = y_submit
submission.to_csv(path_save + 'call2_submit.csv')

#6. PLT
plt.plot(hist.history['acc'], label='acc', color='red')
plt.plot(hist.history['val_acc'], label='val_acc', color='blue')
plt.plot(hist.history['loss'], label='loss', color='green')
plt.plot(hist.history['val_loss'], label='val_loss', color='yellow')
plt.legend()
plt.show()