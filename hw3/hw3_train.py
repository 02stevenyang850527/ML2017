import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, CSVLogger
from keras.preprocessing.image import ImageDataGenerator


num_classes = 7
#--------read train.csv---------
train = pd.read_csv('train.csv')
train_feature = train.feature.str.split(' ')
train_feature = train_feature.tolist()
train_feature = np.array(train_feature, dtype=float)
train_feature = train_feature/255
train_label = np.array(train['label'])

#-------read test.csv------------
'''
test  = pd.read_csv('test.csv')
test_feature = test.feature.str.split(' ')
test_feature = test_feature.tolist()
test_feature = np.array(test_feature, dtype=float)
test_feature = test_feature/255
test_feature = test_feature.reshape(test_feature.shape[0],48,48,1)'''

#--------training & validation data--------
valid_num = 5000
valid_feature = train_feature[:valid_num]
valid_label = train_label[:valid_num]

train_feature = train_feature[valid_num:]
train_label = train_label[valid_num:]

train_feature = train_feature.reshape(train_feature.shape[0],48,48,1)
valid_feature = valid_feature.reshape(valid_feature.shape[0],48,48,1)
train_label = np_utils.to_categorical(train_label,num_classes)
valid_label = np_utils.to_categorical(valid_label,num_classes)

#-------- CNN ------------
model = Sequential()
model.add(Convolution2D(128,(3,3),input_shape=(48,48,1),padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2,padding='same'))
model.add(Dropout(0.25))

model.add(Convolution2D(128,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2,padding='same'))
model.add(Dropout(0.3))

model.add(Convolution2D(256,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2,padding='same'))
model.add(Dropout(0.3))

model.add(Convolution2D(256,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2,padding='same'))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))


model.add(Dense(512))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adamax',metrics=['accuracy'])

#early_stopping = EarlyStopping(monitor = 'val_loss', patience=2)

datagen = ImageDataGenerator(
        featurewise_center=False,               # set input mean to 0 over the dataset
        samplewise_center=False,                # set each sample mean to 0
        featurewise_std_normalization=False,    # divide inputs by std of the dataset
        samplewise_std_normalization=False,     # divide each input by its std
        zca_whitening=False,                    # apply ZCA whitening
        rotation_range=10,                      # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,                  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,                 # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,                   # randomly flip images
        vertical_flip=False)                    # randomly flip images

datagen.fit(train_feature)

batch = 128
'''
bound = 0.9
model.fit(train_feature, train_label,validation_data=(valid_feature,valid_label), batch_size=batch,epochs=5)
prob = model.predict(test_feature)
label = prob.argmax(1)
decision = prob.max(1) > bound
label = label[decision]
feature = test_feature[decision,:,:,:]
test_feature = np.delete(test_feature,decision,0)
train_feature = np.concatenate((train_feature,feature),axis=0)
label = np_utils.to_categorical(label,num_classes)
train_label = np.concatenate((train_label, label), axis=0)'''

model.fit(train_feature, train_label,validation_data=(valid_feature,valid_label), batch_size=batch,epochs=5)
model.fit_generator(datagen.flow(train_feature,train_label,batch_size=batch),
                    steps_per_epoch = train_feature.shape[0]/batch,
                    epochs=100,
                    validation_data = (valid_feature, valid_label)
                   )

'''
for i in range(6):
    model.fit(train_feature, train_label,validation_data=(valid_feature,valid_label), batch_size=batch,epochs=5)
    model.fit_generator(datagen.flow(train_feature,train_label,batch_size=batch),
                        steps_per_epoch = train_feature.shape[0]/batch,
                        epochs=5,
                        validation_data = (valid_feature, valid_label)
                       )
    if (len(test_feature)):
        prob = model.predict(test_feature)
        label = prob.argmax(1)
        decision = prob.max(1) > bound
        label = label[decision]
        feature = test_feature[decision,:,:,:]
        test_feature = np.delete(test_feature,decision,0)
        train_feature = np.concatenate((train_feature,feature),axis=0)
        label = np_utils.to_categorical(label,num_classes)
        train_label = np.concatenate((train_label, label), axis=0)'''


#model.fit(train_feature, train_label,validation_data=(valid_feature,valid_label), batch_size=batch,epochs=10)
score = model.evaluate(train_feature,train_label)

print ('\nTrain Acc: ', score[1])
model.save('my_model.h5')
