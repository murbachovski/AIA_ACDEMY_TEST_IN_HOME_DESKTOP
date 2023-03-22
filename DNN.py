import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

dataset = np.array(range(1, 101))
timeSteps = 5
x_predict = np.array(range(96, 106))

def split_data(dataset, timeSteps):
    data = []
    for i in range(len(dataset) - timeSteps + 1):
        subset = dataset[i : (i + timeSteps)]
        data.append(subset)
    return np.array(data)

train_data = split_data(dataset, timeSteps)
#print(train_data) # 1 ~ 100
#print(train_data.shape) #(96, 5)

x = train_data[:, :-1]
y = train_data[:, -1]
x_predict = split_data(x_predict, 4)
print(x, y, x_predict)
print(x.shape, y.shape, x_predict.shape) # (96, 4) (96,) (7, 4)

#2. MODEL
model = Sequential()
model.add(Dense(32, input_shape=(4,)))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))

#3.COMPILE
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y, epochs=1000, batch_size=32)

#4.PREDICT
loss = model.evaluate(x, y)
result = model.predict(x_predict)
print('loss: ', loss, 'result: ', result)
# [[ 99.98791 ]
#  [100.98762 ]
#  [101.987335]
#  [102.987045]
#  [103.986755]
#  [104.98645 ]
#  [105.98619 ]]