import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import time
#1. DATA
path = 'c:/study_data/_data/dogs_breed/'
save_path = 'c:/study_data/_save/dogs_breed/'

all_data = ImageDataGenerator(
    rescale=1./255
)

start = time.time()
# print(all_data)
# <keras.preprocessing.image.ImageDataGenerator object at 0x0000026E8B790D00>
# Found 1030 images belonging to 5 classes.
dogs_breed_data = all_data.flow_from_directory(
    path,
    target_size=(50, 50),
    batch_size=100,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True
)

# print(dogs_breed_data.image_shape) # (50, 50, 3)
x_dogs_breed = dogs_breed_data[0][0]
y_dogs_breed = dogs_breed_data[0][1]

end = time.time()
# print(x_dogs_breed)
# print(x_dogs_breed.shape) # (100, 50, 50, 3)
# print(y_dogs_breed)
# print(y_dogs_breed.shape) # (100, 5)

x_train, x_test, y_train, y_test = train_test_split(
    x_dogs_breed,
    y_dogs_breed,
    test_size=0.25,
    random_state=1111
)

augment_size = 100

# print(x_train.shape)        # (75, 50, 50, 3)
# print(x_train[0].shape)     # (50, 50, 3)
# print(x_train[1].shape)     # (50, 50, 3)
# print(x_train[0][0].shape)  # (50, 3)

dogs_breed_data = all_data.flow(
    np.tile(x_train[0].reshape(50*50, 3), augment_size).reshape(-1, 50, 50, 3),
    np.zeros(augment_size),
    batch_size=augment_size,
    shuffle=True
)

print(x_train.shape, x_test.shape) # (75, 50, 50, 3) (25, 50, 50, 3)
print(y_train.shape, y_test.shape) # (75, 5) (25, 5)

np.save(save_path + 'keras56_dogs_breed_aru_x_train.npy', arr=x_train)
np.save(save_path + 'keras56_dogs_breed_aru_x_test.npy', arr=x_test)
np.save(save_path + 'keras56_dogs_breed_aru_y_train.npy', arr=y_train)
np.save(save_path + 'keras56_dogs_breed_aru_y_test.npy', arr=y_test)
