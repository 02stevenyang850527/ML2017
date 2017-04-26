import os
import sys
import numpy as np
import pandas as pd
import itertools
from sklearn.metrics import confusion_matrix
from keras.models import load_model
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

model = load_model('my_model.h5')
train = pd.read_csv('train.csv')
train_feature = train.feature.str.split(' ')
train_feature = train_feature.tolist()
train_feature = np.array(train_feature, dtype=float)
train_feature = train_feature/255
train_label = np.array(train['label'])
valid_num = 5000
valid_feature = train_feature[:valid_num]
valid_label = train_label[:valid_num]
valid_feature = valid_feature.reshape(valid_feature.shape[0],48,48,1)

np.set_printoptions(precision=2)
predictions = model.predict_classes(valid_feature)
conf_mat = confusion_matrix(valid_label,predictions)

plt.figure()
plot_confusion_matrix(conf_mat, classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"])
plt.show()

