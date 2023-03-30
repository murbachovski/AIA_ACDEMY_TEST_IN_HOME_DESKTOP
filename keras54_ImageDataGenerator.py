import numpy as np
from keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(
    rescale=1./255, # MinMaxScaler, 정규화, 전처리, Nomalization
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(
    rescale=1./255 # 평가 데이터이므로 증폭X
)

xy_train = train_datagen.flow_from_directory(
    'c:/study_data/_data/brain/train/',
    target_size=(100, 100), # 각각 다른 사이즈를 가진 이미지들을 (200, 200)사이즈로 맞춰 준다.
    batch_size=160,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True
)

xy_test = test_datagen.flow_from_directory(
    'c:/study_data/_data/brain/test/',
    target_size=(100, 100), # 각각 다른 사이즈를 가진 이미지들을 (200, 200)사이즈로 맞춰 준다.
    batch_size=160,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True
)

# print(xy_train)
# print(xy_test)
# <keras.preprocessing.image.DirectoryIterator object at 0x000001E4C05F0A00>
# <keras.preprocessing.image.DirectoryIterator object at 0x000001E4C07B8FA0>
# print(len(xy_train)) # 32
# print(len(xy_train[0][1])) 
# print(xy_train[0][0])
# print(xy_train[0][1])
# print(xy_train[0][0].shape)
# print(xy_train[0][1].shape)


#2. MODEL
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten
model = Sequential()
model.add(Conv2D(64, (2, 2), input_shape=(100, 100, 1), activation='relu')) # 왜 (100, 100, 1)이 들어오지?
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. COMPILE
from tensorflow.python.keras.callbacks import EarlyStopping
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
Es = EarlyStopping(
    monitor='val_acc',
    patience=20,
    mode='max',
    restore_best_weights=True
)
hist = model.fit_generator(xy_train,
                           epochs=100,
                           steps_per_epoch=1, # 전체 데이터/batch = 160/5 = 32
                           validation_data=xy_test,
                           validation_steps=1 # validation_data/batch = 120/5 = 24
                           )
loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']
print('loss: ', loss[-1], 'val_loss: ', val_loss[-1], 'acc: ', acc[-1], 'val_acc: ', val_acc[-1])

from matplotlib import pyplot as plt
plt.subplot(1,2,1)
plt.plot(range(len(hist.history['loss'])),hist.history['loss'],label='loss')
plt.plot(range(len(hist.history['val_loss'])),hist.history['val_loss'],label='val_loss')
plt.legend()
plt.subplot(1,2,2)
plt.plot(range(len(hist.history['acc'])),hist.history['acc'],label='acc')
plt.plot(range(len(hist.history['val_acc'])),hist.history['val_acc'],label='val_acc')
plt.legend()
plt.show()