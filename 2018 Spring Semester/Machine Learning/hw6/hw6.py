import csv
import numpy as np
import sys
from keras import regularizers
from keras.layers import Input, Embedding, Dense, BatchNormalization, Flatten, dot, add, merge
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn import  preprocessing
import pandas as pd

n_users = 6041
n_movies = 3953
n_latent_factors = 7

# Preprocessing
test = pd.read_csv(sys.argv[1], names=["id", "user_id", "movie_id"], header=0)
movies = pd.read_csv(sys.argv[3], names=['movie_id', 'title', 'genres'], delimiter='::', header=0)
users = pd.read_csv(sys.argv[4], names=['user_id', 'gender', 'age', 'occupation', 'zip_code'], delimiter='::', header=0)


# Model Struture
inputs_movies = Input(shape=(1,))
inputs_users = Input(shape=(1,))

embedding_movies = Embedding(n_movies, n_latent_factors, embeddings_initializer='random_normal', embeddings_regularizer=regularizers.l2(1e-6))(inputs_movies)
embedding_users = Embedding(n_users, n_latent_factors, embeddings_initializer='random_normal', embeddings_regularizer=regularizers.l2(1e-6))(inputs_users)

flatten_movies = Flatten()(embedding_movies)
flatten_users = Flatten()(embedding_users)

bias_movies = Embedding(n_movies, 1)(inputs_movies)
bias_movies = Flatten()(bias_movies)
bias_movies = Dense(1, activation='linear')(bias_movies)
bias_users = Embedding(n_users, 1)(inputs_users)
bias_users = Flatten()(bias_users)
bias_users = Dense(1, activation='linear')(bias_users)

doted = dot([flatten_movies, flatten_users], 1)
doted = Dense(1, activation='linear')(doted)

added = add([doted, bias_movies, bias_users])
DNN_inputs = Dense(128, activation='linear')(added)
DNN_inputs = Dense(64, activation='linear')(DNN_inputs)
DNN_inputs = Dense(32, activation='linear')(DNN_inputs)
outputs = Dense(1)(DNN_inputs)
# model = Model([inputs_users,inputs_movies], outputs)
# model.compile('adam', loss='mean_squared_error')
#model.summary()

model = load_model(sys.argv[5])
model.summary()
pred = model .predict([test.user_id, test.movie_id], verbose=1).squeeze()
pred = pred * 4 + 1
# pred = np.clip(pred, None, None)
result = [[i+1, '{0:.2f}'.format(pred[i])] for i in range(len(pred))]
df = pd.DataFrame(result, columns = ["TestDataID", "Rating"])
df.to_csv(sys.argv[2], index=False)