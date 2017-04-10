import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils


num_classes = 7
#--------read train.csv---------
train = pd.read_csv('train.csv')
train_feature = train.feature.str.split(' ')
train_feature = train_feature.tolist()
train_feature = np.array(train_feature, dtype=float)
train_feature = train_feature/255
train_label = np.array(train['label'])

#--------training & validation data--------
#valid_amount = 5000
#valid_feature = train_feature[:valid_amount]
#valid_label = train_label[:valid_amount]

#train_feature = np.delete(train_feature,range(valid_amount),axis=0)
#train_label = np.delete(train_label,range(valid_amount),axis=0)
train_feature = train_feature.reshape(train_feature.shape[0],48,48,1)
#valid_feature = valid_feature.reshape(valid_amount,48,48,1)

train_label = np_utils.to_categorical(train_label,num_classes)
#valid_label = np_utils.to_categorical(valid_label,num_classes)

#-------- CNN ------------
model = Sequential()
model.add(Convolution2D(43,(5,5),input_shape=(48,48,1)))
model.add(MaxPooling2D(3,3))
model.add(Convolution2D(80,(5,5)))
model.add(MaxPooling2D(3,3))
model.add(Flatten())
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

#model.fit(train_feature, train_label, validation_data=(valid_feature, valid_label), batch_size=30,epochs=10)
model.fit(train_feature, train_label, validation_split = 0.2, batch_size=30,epochs=10)

score = model.evaluate(train_feature,train_label,batch_size=10000)
print ('Train Acc: ', score[1])

model.save('my_model.h5')
