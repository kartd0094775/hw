import numpy as np
import csv, time
import sys
import os
from skimage import io, transform
from skimage.color import rgb2gray, gray2rgb

from keras.utils import np_utils
from keras.models import Sequential
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers import Conv2D, MaxPooling2D, Flatten, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
import tensorflow as tf, keras
import keras.backend as K
# import keras.backend.tensorflow_backend as KTF
# from sklearn.model_selection import train_test_split

# def get_session(gpu_fraction=0.8):

#    num_threads = os.environ.get('OMP_NUM_THREADS')
#    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

#    if num_threads:
#       return tf.Session(config=tf.ConfigProto(
#          gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
#    else:
#       return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# os.environ["THEANO_FLAGS"] = "device=gpu0"
# KTF.set_session(get_session())
def load_testing_images(path):
    images = []
    filenames = []
    # for filename in os.listdir(path):
    #     if filename not in dict_img2id.keys():
    #         images.append(io.imread(path+'/'+filename))
    #         filenames.append(filename)

    # test_dir = sorted(os.listdir('data/new_box/test_gray'))
    test_dir = sorted(os.listdir('image/bounding3/test'))
    count=0
    for filename in test_dir:
        count+=1
        if count==10:
            pass
        img = io.imread(path+'/'+filename)
        # img = transform.resize(img, output_shape=(160, 160), mode='reflect')
        # if len(img.shape) == 2:
        #   img = gray2rgb(img)
        images.append(img)
        filenames.append(filename)
    for idx in range(0, len(images)):
      images[idx] = transform.resize(images[idx], output_shape=(160, 160), mode='reflect')
      if len(images[idx].shape) == 2:
        images[idx] = gray2rgb(images[idx])
    images = np.array(images)
    images = images/255.0
    return images, filenames

def load_training_images(path):
    images = []
    labels = []
    count = 0
    for filename in os.listdir(path):
        if filename in dict_img2id.keys():
        #(io.imread(dir+'/'+filename)).flatten()np.concatenate(images, (io.imread(dir+'/'+filename)).flatten())
            img = io.imread(path+'/'+filename)
            # img = transform.resize(img, output_shape=(160, 160), mode='reflect')
            # if len(img.shape) == 2:
            #   img = gray2rgb(img)
            images.append(img)
            labels.append(dict_img2id[filename])
            count+=1
        if count==50:
          pass
    for idx in range(0, len(images)):
      images[idx] = transform.resize(images[idx], output_shape=(160, 160), mode='reflect')
      if len(images[idx].shape) == 2:
        images[idx] = gray2rgb(images[idx])
    images = np.array(images)
    labels = np.array(labels, dtype=int)
    images = images/255
    filename = None
    return images, labels
def mapK(y_true, y_pred):
  K = 5; precisions = []; p = 0
  # y_sorted = np.array(y_pred).argsort()[-K:][::-1]
  y_pred_top_k, y_pred_ind_k = tf.nn.top_k(y_pred, K)
  y_true_top_k, y_true_ind_k = tf.nn.top_k(y_true, 1)
  for j in range(0, K):
    temp = tf.equal(y_pred_ind_k[:, j], y_true_ind_k[:, 0])
    temp = tf.cast(temp, tf.float64)
    p = tf.reduce_mean(temp)
    precisions.append(p/(j+1))
  return keras.backend.sum(precisions)

with open('data/table_ID_number.csv') as file:
    data = csv.reader(file)
    next(data, None)
    dict_id2idnum = {}
    dict_idnum2id = {}
    for line in data:
        dict_id2idnum[line[0]] = line[1]
        dict_idnum2id[line[1]] = line[0]

with open('data/train_with_number.csv') as file:
    data = csv.reader(file)
    next(data, None)
    dict_img2id = {}
    for line in data:
        dict_img2id[line[0]] = line[2]

# img_dir = './data/new_box/train_gray'
img_dir = 'image/bounding3/train'
X, Y = load_training_images(img_dir)
test, filenames = load_testing_images('image/bounding3/test')
test = test.reshape(test.shape[0], test.shape[1], test.shape[2], 3)

# remove new whale
print('remove new whale')
new_whale_idx = np.where(Y==7)[0]
X = np.delete(X, new_whale_idx, axis=0)
Y = np.delete(Y, new_whale_idx, axis=0)
new_whale_idx = None

# no 7 now
over_new_whale_idx = np.where(Y>7)[0]
Y[over_new_whale_idx] -= 1
over_new_whale_idx = None

X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 3)
Y = np_utils.to_categorical(Y, 4250)
# Y = np_utils.to_categorical(Y, 4251)

for i in range(9, 10):
  # print('get validation data')
  valid_percentage = 0
  valid_len = int(X.shape[0]*valid_percentage)
  bagging_idx = np.random.permutation(list(range(X.shape[0])))
  train_idx = bagging_idx[valid_len:]
  valid_idx = bagging_idx[:valid_len]
  train_data = X[train_idx]
  train_label = Y[train_idx]
  valid_data = X[valid_idx]
  valid_label = Y[valid_idx]
  bagging_idx = None
  # train_data, train_label, valid_data, valid_label = train_test_split(X, Y, test_size=0.1)
  # X=None; Y=None
  # print(X.shape, Y.shape)
  datagen = ImageDataGenerator(horizontal_flip=True,\
                                  rotation_range=10, \
                                  # shear_range = 0.3, zoom_range=(1.5, 0.7),\
                                  width_shift_range=0.1,height_shift_range=0.1)
  # datagen = ImageDataGenerator()
  datagen.fit(train_data)
  # first training
  print('start training')
  dropout_rate = 0.6
  mobilenet = keras.applications.mobilenet.MobileNet(input_shape=(160, 160, 3), alpha=1.0, depth_multiplier=1,
                                                dropout=1e-3, include_top=False, weights='imagenet',
                                                input_tensor=None, pooling='max', classes=4250)

  model = Sequential()
  model.add(mobilenet)
  model.add(Dense(units=512, kernel_initializer='glorot_normal'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(0.6))

  model.add(Dense(units=512, kernel_initializer='glorot_normal'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(0.6))

  model.add(Dense(units=4250))
  model.add(BatchNormalization())
  model.add(Activation('softmax'))

  adam = Adam(lr=0.01, decay=0.01/200)
  model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['top_k_categorical_accuracy', mapK])


  history = model.fit_generator(datagen.flow(train_data,train_label,batch_size=20, shuffle=True), \
                      steps_per_epoch=train_data.shape[0]/32,epochs = 700,\
                      #validation_data=(valid_data,valid_label), \
                      #callbacks=callbacks_list,
                      verbose=1
                      )
  print('predicting...')
  result = model.predict(test,32, verbose=1)

  # insert 7 between 6 and 8 (new whale)
  left_result = result[:, :7]
  right_result = result[:, 7:]
  result = np.concatenate([left_result, np.zeros((result.shape[0], 1)), right_result], axis=1)
  left_result = None
  right_result = None


  print('writing predict.csv')
  with open('predict/predict.csv','w',newline='') as file:
      writer = csv.writer(file)
      writer.writerow(['Image','Id'])
      count = 0
      for res in result:
          predict_prob_sort = sorted(res, reverse=True)
          predict_idnum_sort = sorted(range(len(res)), key=lambda k: res[k], reverse=True)
          output = ''
          for i in range(20):
            output += dict_idnum2id[str(predict_idnum_sort[i])] + ' '
          writer.writerow([filenames[count], output])
          count += 1
  # release memory
  history= None
  model = None
  result = None
  K.clear_session()

print('program ends')
