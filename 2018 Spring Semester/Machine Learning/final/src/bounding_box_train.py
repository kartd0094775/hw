import argparse
import logging
import os

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import (Activation, BatchNormalization, Conv2D, Dense,
                          Dropout, Flatten, MaxPooling2D)
from keras.models import Sequential
from keras.preprocessing.image import array_to_img, img_to_array
from numpy.linalg import inv
from PIL import Image as pil_image
from scipy.ndimage import affine_transform
from skimage.color import gray2rgb
from skimage.transform import resize
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='ML final: Train Bounding Box Model for Humpback Whale Identification Challenge')
    # Path arguments
    parser.add_argument('label_data_path')
    parser.add_argument('training_img_dir')
    parser.add_argument('model_save_dir')
    # Model arguments
    parser.add_argument('--img_size', default=(128, 256), type=tuple)
    parser.add_argument('--max_rotation_angle', default=10, type=int)
    parser.add_argument('--rotation_step', default=5, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--val_frac', default=0.1, type=float)
    
    args = parser.parse_args()
    return args


def ensure_dir(filepath):
    directory = os.path.dirname(filepath)
    if len(directory) > 0 and not os.path.exists(directory):
        os.makedirs(directory)


def load_label_data(path):
    """Load label data from Martin Piotte
    Reference: https://www.kaggle.com/martinpiotte/bounding-box-data-for-the-whale-flukes
    """
    logging.info('Load label data')
    with open(path, 'rt') as f:
        data = f.read().split('\n')[:-1]
    data = [line.split(',') for line in data]
    data = [(p, [(int(coord[i]), int(coord[i+1]))
                 for i in range(0, len(coord), 2)]) for p, *coord in data]
    return data


def read_raw_image(training_img_dir, p):
    return pil_image.open(os.path.join(training_img_dir, p))


def bounding_rectangle(list):
    x0, y0 = list[0]
    x1, y1 = x0, y0
    for x, y in list[1:]:
        x0 = min(x0, x)
        y0 = min(y0, y)
        x1 = max(x1, x)
        y1 = max(y1, y)
    return x0, y0, x1, y1


def transform_img(x, affine):
    matrix = affine[:2, :2]
    offset = affine[:2, 2]
    x = np.moveaxis(img_to_array(x), -1, 0)  # Change to channel first
    channels = [affine_transform(channel, matrix, offset, order=1,
                                 mode='constant', cval=np.average(channel)) for channel in x]
    # Back to channel last, and image format
    return array_to_img(np.moveaxis(np.stack(channels, axis=0), 0, -1))


def coord_transform(coordinates, m):
    result = []
    for x, y in coordinates:
        y, x, _ = m.dot([y, x, 1]).astype(np.int)
        result.append((x, y))
    return result


def load_training_data(label_data, training_img_dir, max_rotation_angle, rotation_step, img_size):
    logging.info('Load training data')
    # No augmentation setting
    if max_rotation_angle == 0:
        rotation_step = 1

    X = []
    y = []

    for filename, coordinates in tqdm(label_data):
        img_raw = read_raw_image(training_img_dir, filename)
        width, height = img_raw.size

        # Augmentation by affine
        for angle in range(-max_rotation_angle, max_rotation_angle+1, rotation_step):
            rotation = np.deg2rad(angle)
            # Place the origin at the center of the image
            center = np.array([[1, 0, -height/2], [0, 1, -width/2], [0, 0, 1]])
            # Rotate
            rotate = np.array([[np.cos(rotation), np.sin(
                rotation), 0], [-np.sin(rotation), np.cos(rotation), 0], [0, 0, 1]])
            # Restaure the origin
            decenter = inv(center)
            # Combine the transformations into one
            m = np.dot(decenter, np.dot(rotate, center))
            img = transform_img(img_raw, m)
            transformed_coordinates = coord_transform(coordinates, inv(m))
            box = bounding_rectangle(transformed_coordinates)

            # Normalize
            img = img_to_array(img) / 255.
            img.astype(np.float16)  # Save memory usage
            # gray2rgb
            if img.shape[2] == 1:
                img = gray2rgb(img.squeeze())
            # resize
            img = resize(img, img_size)
            X.append(img)

            box = [box[1], box[0], box[3], box[2]]
            box[0], box[2] = float(box[0]) / height, float(box[2]) / height
            box[1], box[3] = float(box[1]) / width, float(box[3]) / width
            y.append(box)

    X = np.array(X, dtype=np.float16)
    y = np.array(y, dtype=np.float32)
    y = np.clip(y, 0., 1.)

    return X, y


def build_model(img_size):
    logging.info('Building model')
    model = Sequential()
    model.add(Conv2D(16, (3, 3), input_shape=img_size+(3,),
                     kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(64, (3, 3), kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(64, (3, 3), kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(64, (3, 3), kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(64, (3, 3), kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(2, 2))

    model.add(Flatten())
    model.add(Dense(units=256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(units=128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(units=4))
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    model.summary()

    return model


def main():
    label_data = load_label_data(args.label_data_path)
    X, y = load_training_data(label_data, args.training_img_dir,
                              args.max_rotation_angle, args.rotation_step, args.img_size)

    model = build_model(args.img_size)
    ensure_dir(args.model_save_dir)
    checkpoint = ModelCheckpoint(os.path.join(args.model_save_dir, '{epoch:02d}_{val_loss:.5f}.h5'),
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True)
    callbacks = [checkpoint]
    model.fit(X, y, epochs=args.epochs, batch_size=args.batch_size,
              validation_split=args.val_frac, shuffle=True, callbacks=callbacks)


if __name__ == '__main__':
    # Set logging config
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    main()
