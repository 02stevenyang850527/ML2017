import numpy as np
import pandas as pd
import keras
import keras.backend as K
from keras.layers import Input, Flatten, Embedding, Dropout, Concatenate, Dot, Add, Dense
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint


def rmse(y_true,y_pred):
    return K.sqrt(K.mean((y_pred - y_true)**2))

def get_model(n_users, n_items, latent_dim = 10):
    user_input = Input(shape=[1])
    item_input = Input(shape=[1])
    user_vec = Embedding(n_users, latent_dim, embeddings_initializer='random_normal')(user_input)
    user_vec = Flatten()(user_vec)
    item_vec = Embedding(n_items, latent_dim, embeddings_initializer='random_normal')(item_input)
    item_vec = Flatten()(item_vec)
    user_bias = Embedding(n_users, 1,embeddings_initializer='zeros')(user_input)
    user_bias = Flatten()(user_bias)
    item_bias = Embedding(n_items, 1,embeddings_initializer='zeros')(item_input)
    item_bias = Flatten()(item_bias)
    r_hat = Dot(axes=1)([user_vec,item_vec])
    r_hat = Add()([r_hat, user_bias, item_bias])
    model = keras.models.Model([user_input, item_input], r_hat)
    model.compile(loss='mse', optimizer='adam',metrics=[rmse])
    return model
    

def nn_model(n_users, n_items,latent_dim=100):
    user_input = Input(shape=[1])
    item_input = Input(shape=[1])
    user_vec = Embedding(n_users, latent_dim, embeddings_initializer='random_normal')(user_input)
    user_vec = Flatten()(user_vec)
    item_vec = Embedding(n_items, latent_dim, embeddings_initializer='random_normal')(item_input)
    item_vec = Flatten()(item_vec)
    merge_vec = Concatenate()([user_vec, item_vec])
    hidden = Dense(150, activation='relu')(merge_vec)
    hidden = BatchNormalization()(hidden)
    hidden = Dropout(0.3)(hidden)
    hidden = Dense(50, activation='relu')(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Dropout(0.3)(hidden)
    output = Dense(1)(hidden)
    model  = keras.models.Model([user_input, item_input], output)
    model.compile(loss='mse', optimizer='adam',metrics=[rmse])
    model.summary()
    return model


def split_data(movie,user,Y,split_ratio=0.1):
    indices = np.arange(movie.shape[0])  
    np.random.shuffle(indices) 
    
    movie_data = movie[indices]
    user_data = user[indices]
    Y_data = Y[indices]
    
    num_validation_sample = int(split_ratio * movie_data.shape[0] )
    
    movie_train = movie_data[num_validation_sample:]
    user_train = user_data[num_validation_sample:]
    Y_train = Y_data[num_validation_sample:]

    movie_val = movie_data[:num_validation_sample]
    user_val = user_data[:num_validation_sample]
    Y_val = Y_data[:num_validation_sample]

    return (user_train,movie_train,Y_train),(user_val,movie_val,Y_val)


'''
users = pd.read_csv('users.csv', sep='::',engine='python')
movies = pd.read_csv('movies.csv', sep='::', engine = 'python')
movies['Genres'] = movies.Genres.str.split('|')

users.Age = users.Age.astype('category')
users.Gender = users.Gender.astype('category')
users.Occupation = users.Occupation.astype('category')
'''

np.random.seed(5)
ratings = pd.read_csv('train.csv')

n_movie = np.max(ratings['MovieID']) + 1
n_users = np.max(ratings['UserID'])  + 1
y = np.zeros((n_users,n_movie))

user = ratings['UserID'].values
movie = ratings['MovieID'].values

Y = ratings['Rating'].values
'''
mean = Y.mean()
std  = Y.std()
Y = (Y-mean)/std
'''
(user_train, movie_train,Y_train),(user_val,movie_val,Y_val) = split_data(movie,user,Y)
model = get_model(n_users, n_movie,10)
earlystopping = EarlyStopping(monitor='val_rmse', patience = 10, verbose=1, mode='max')

model.fit([user_train, movie_train],Y_train,epochs=400, batch_size=128, validation_data=([user_val,movie_val],Y_val),callbacks=[earlystopping])

model.save('mf.h5')
