from keras.preprocessing.text import Tokenizer
import numpy as np
from sklearn.preprocessing import OneHotEncoder

text = '나는 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다.'

# 수치화
token = Tokenizer()
token.fit_on_texts([text])
# print(token.word_counts)
# print(token.analyzer)
# print(token.filters)
# print(token.analyzer)
# print(token.word_index) # {'마구': 1, '매우': 2, '나는': 3, '진짜': 4, '맛있는': 5, '밥을': 6, '엄청': 7, '먹었다': 8}
x = token.texts_to_sequences([text])
# print(x) [[3, 4, 2, 2, 5, 6, 7, 1, 1, 1, 8]]

# 1. to_categorical
from keras.utils import to_categorical
# x = to_categorical(x)
# print(x)
# [[[0. 0. 0. 1. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 1. 0. 0. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 1. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 1. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 1. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 1.]]]

# 2. get_dummies
import pandas as pd
# x = pd.get_dummies(np.array(x).reshape(11,)) # => Only 1Tensor
# x = pd.get_dummies(np.array(x).ravel()) => It's make a 1Tensor
# print(x)
#     1  2  3  4  5  6  7  8
# 0   0  0  1  0  0  0  0  0
# 1   0  0  0  1  0  0  0  0
# 2   0  1  0  0  0  0  0  0
# 3   0  1  0  0  0  0  0  0
# 4   0  0  0  0  1  0  0  0
# 5   0  0  0  0  0  1  0  0
# 6   0  0  0  0  0  0  1  0
# 7   1  0  0  0  0  0  0  0
# 8   1  0  0  0  0  0  0  0
# 9   1  0  0  0  0  0  0  0
# 10  0  0  0  0  0  0  0  1

# 3. skelearn_OneHot
ohe = OneHotEncoder() # => Only 2Tensor
x = ohe.fit_transform(np.array(x).reshape(-1, 1)).toarray()
# print(x)
# [[0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 1. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 1. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 1.]]
#복습
