import os, os.path


import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

num_train =0
num_val = 0
batch_size = 64
num_epoch = 2

current_dir = os.getcwd()
data_dir =os.path.join(os.path.dirname(current_dir), 'resources/data')

if os.path.isdir(data_dir):
    train_dir = os.path.join(data_dir,'train')
    val_dir = os.path.join(data_dir,'test')

    list_train_sub_dirs = os.listdir(train_dir)
    list_val_sub_dirs = os.listdir(val_dir)
    for i in list_train_sub_dirs:
        num_train = num_train +len(os.listdir(os.path.join(train_dir,i)))
        num_val = num_val+len(os.listdir(os.path.join(val_dir,i)))


train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')


model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])
model_info = model.fit_generator(
          train_generator,
          steps_per_epoch=num_train // batch_size,
          epochs=num_epoch,
          validation_data=validation_generator,
          validation_steps=num_val // batch_size)
model.save_weights('test_model.h5')