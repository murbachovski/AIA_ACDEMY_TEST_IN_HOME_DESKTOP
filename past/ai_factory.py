import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, Flatten
#1. DATA
path = './_data/AIFac_air/'
path_save = './_save/AIFac_air/'

train_csv = pd.read_csv(path + 'train_data.csv', index_col=0)
test_csv = pd.read_csv(path + 'test_data.csv', index_col=0)

# print(train_csv, test_csv)
# print(train_csv.shape, test_csv.shape) # (2463, 8) (7389, 8)
# 테스트 데이터가 더 많네

# ISNULL
train_csv = train_csv.dropna()
# print(train_csv.shape)

# x, y SPLIT
x = train_csv.drop(['type'], axis=1)
y = train_csv['type']
# print(x, y)
# print(x.shape, y.shape) # (2463, 6) (2463,)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.25,
    random_state=2222,
    stratify=y
)
# print(x_train, x_test)
# print(y_train, y_test)
# print(x_train.shpae, x_test.shape)
# print(y_train.shape, y_test.shape)

#2. MODEL
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(7, )))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

#3. COMPILE
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
model.fit(x_test, y_test, epochs=500, batch_size=50)

#4. PREDICT
results = model.evaluate(x_test)
print(results)

#5. SUBMIT
y_submit = model.predict(test_csv)
submission = pd.read_csv(path + 'answer_sample.csv', index_col=0)
submission['type'] = y_submit
submission.to_csv(path_save + 'ai_factoty_submit.csv')