import pandas as pd
import numpy as np
import sys
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
train = pd.read_csv(sys.argv[1])
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
test_feature = test_feature.reshape(test_feature.shape[0],48,48,1)
'''
#--------training & validation data--------
valid_num = 5000
valid_feature = train_feature[:valid_num]
valid_label = train_label[:valid_num]

train_feature = train_feature[valid_num:]
train_label = train_label[valid_num:]

#train_feature = train_feature.reshape(train_feature.shape[0],48,48,1)
#valid_feature = valid_feature.reshape(valid_feature.shape[0],48,48,1)
train_label = np_utils.to_categorical(train_label,num_classes)
valid_label = np_utils.to_categorical(valid_label,num_classes)

#-------- CNN ------------
model = Sequential()
model.add(Dense(input_dim=48*48,units=256,activation='relu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Dense(units=512,activation='relu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Dense(units=640,activation='relu'))
model.add(Dropout(0.4))
model.add(BatchNormalization())
model.add(Dense(units=1024,activation='relu'))
model.add(Dropout(0.4))
model.add(BatchNormalization())
model.add(Dense(units=num_classes,activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',optimizer='adamax',metrics=['accuracy'])
'''
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
'''
csv_logger0 = CSVLogger('training_dnn0.csv')
#csv_logger1 = CSVLogger('training_dnn1.csv')
batch = 128
model.fit(train_feature, train_label,validation_data=(valid_feature,valid_label), batch_size=batch,epochs=105,callbacks=[csv_logger0])
'''
model.fit_generator(datagen.flow(train_feature,train_label,batch_size=batch),
                    steps_per_epoch = train_feature.shape[0]/batch,
                    epochs=100,
                    validation_data = (valid_feature, valid_label),
                    callbacks = [csv_logger1]
                   )

'''
score = model.evaluate(train_feature,train_label)

print ('\nTrain Acc: ', score[1])
model.save('my_model_dnn.h5')

