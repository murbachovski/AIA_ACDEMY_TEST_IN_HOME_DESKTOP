from keras.preprocessing.text import Tokenizer
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, LSTM, Reshape


#1. DATA
docs = ['너무 재밋어요', '참 최고에요', '참 잘 만든 영화예요',
        '추천하고 싶은 영화입니다.', '한 번 더 보고 싶네요.', '글세요',
        '별로에요', '생각보다 지루해요', '연기가 어색해요',
        '재미없어요', '너무 재미없다', '참 재밋네요', '환희가 잘 생기긴 했어요',
        '환희가 안해요'
        ]

# 긍정1, 부정0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1,0])

# 수치화
token = Tokenizer()
token.fit_on_texts(docs)
x = token.fit_on_sequences(docs)
y = labels

from keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre', maxlen=5)
word_size = len(token.word_index)
print('단어 사전 갯수: ', word_size)

#2. MODEL
model = Sequential()
model.add(Reshape(target_shape=(5, 1), input_shape=(5,)))
model.add(LSTM(32))
model.add(Dense(10))
model.add(Dense(1, activation='sigmoid'))

#3. COMPILE
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(pad_x, labels, epochs=10, batch_size=10)

#4. PREDICT
acc = model.evaluate(pad_x, labels)[1]
print('acc: ', acc)


