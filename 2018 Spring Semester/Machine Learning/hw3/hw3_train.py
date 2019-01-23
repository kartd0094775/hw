import sys
import csv
import tensorflow as tf
import numpy as np
from PIL import Image
from keras.callbacks import *
from keras.models import Sequential, load_model
from keras.layers import ZeroPadding2D, BatchNormalization, Conv2D, PReLU, MaxPooling2D, Flatten
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.preprocessing import image
from keras.callbacks import *

train_csv = sys.argv[1]

text1 = open(train_csv, 'r')

train = csv.reader(text1, delimiter=',')

def preprocessing(train):
    train_x = []
    train_y = []
    
    counter = 0
    for row in train:
        if counter == 0: 
            counter += 1
            continue
        train_y.append(np_utils.to_categorical(int(row[0]), 7)[0])
        temp = []
        for pixel in row[1].split():
            temp.append(int(pixel))
        temp = np.array(temp).reshape(48, 48, 1)
        train_x.append(temp)
        counter += 1
    train_x = np.array(train_x)
    train_y = np.array(train_y)
       
    return train_x, train_y

def aug_flip_right_left(train_x, train_y, num_class=None):
    aug_x = []
    aug_y = []
    if num_class is None:
        for i in range(len(train_x)):
            img = image.array_to_img(train_x[i]).transpose(Image.FLIP_LEFT_RIGHT)
            aug_x.append(np.array(image.img_to_array(img)))
            aug_y.append(train_y[i])
    else:
        for i in range(len(train_x)):
            if train_y[i].argmax() == num_class:
                img = image.array_to_img(train_x[i]).transpose(Image.FLIP_LEFT_RIGHT)
                aug_x.append(np.array(image.img_to_array(img)))
                aug_y.append(train_y[i])
    return np.array(aug_x), np.array(aug_y)
def normalization(train_x):
    mean = np.load('mean.npy')
    std = np.load('std.npy')
    return np.divide(train_x - mean, std)

def validation(x, y):
    index = x.shape[0]//10
    validate_x = x[:index]
    validate_y = y[:index]
    train_x = x[index:]
    train_y = y[index:]
    return validate_x, validate_y, train_x, train_y


if __name__ == '__main__':
    train_x, train_y= preprocessing(train)
    train_x = normalization(train_x)
    validate_x, validate_y, train_x, train_y = validation(train_x, train_y)
    merged_x, merged_y = aug_flip_right_left(train_x, train_y)
    np.random.seed(6666)
    x = np.random.permutation(np.concatenate((train_x, merged_x), axis=0))
    np.random.seed(6666)
    y = np.random.permutation(np.concatenate((train_y, merged_y), axis=0))

    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(48, 48, 1)))
    model.add(Conv2D(64, (3, 3), kernel_initializer='glorot_normal') )
    model.add( BatchNormalization() )
    model.add( PReLU() )
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (3, 3), kernel_initializer='glorot_normal') )
    model.add( BatchNormalization() )
    model.add( PReLU() )
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), kernel_initializer='glorot_normal') )
    model.add( BatchNormalization() )
    model.add( PReLU() )
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), kernel_initializer='glorot_normal') )
    model.add( BatchNormalization() )
    model.add( PReLU() )
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), kernel_initializer='glorot_normal') )
    model.add( BatchNormalization() )
    model.add( PReLU() )
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), kernel_initializer='glorot_normal') )
    model.add( BatchNormalization() )
    model.add( PReLU() )
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), kernel_initializer='glorot_normal') )
    model.add( BatchNormalization() )
    model.add( PReLU() )
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), kernel_initializer='glorot_normal') )
    model.add( BatchNormalization() )
    model.add( PReLU() )
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), kernel_initializer='glorot_normal') )
    model.add( BatchNormalization() )
    model.add( PReLU() )
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), kernel_initializer='glorot_normal') )
    model.add( BatchNormalization() )
    model.add( PReLU() )
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), kernel_initializer='glorot_normal') )
    model.add( BatchNormalization() )
    model.add( PReLU() )
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), kernel_initializer='glorot_normal') )
    model.add( BatchNormalization() )
    model.add( PReLU() )
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), kernel_initializer='glorot_normal') )
    model.add( BatchNormalization() )
    model.add( PReLU() )
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add (Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(4096, kernel_initializer='glorot_normal'))
    model.add( PReLU() )
    model.add(Dropout(0.5))
    model.add(Dense(4096, kernel_initializer='glorot_normal'))
    model.add( PReLU() )
    model.add(Dropout(0.5))
    model.add( Dense(units=7) )
    model.add( Activation('softmax') )
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    datagen = image.ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
        ) 
    modelcheck = ModelCheckpoint('./public.h5', monitor='val_acc', verbose=1, save_best_only=True)
    earlystop = EarlyStopping(monitor='loss', patience=30)
    datagen.fit(train_x)
    model.fit_generator(datagen.flow(train_x, train_y, batch_size=16), steps_per_epoch=1024, epochs=500, validation_data=(validate_x, validate_y), callbacks=[modelcheck, earlystop])
        