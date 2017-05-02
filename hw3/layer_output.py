import matplotlib.pyplot as plt
import sys
import pandas as pd
from keras.models import load_model
from keras import backend as K
import numpy as np


emotion_classifier = load_model('my_model.h5')
layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[1:])

input_img = emotion_classifier.input
name_ls = ['activation_1', 'activation_2','activation_3','activation_4']
collect_layers = [ K.function([input_img, K.learning_phase()], [layer_dict[name].output]) for name in name_ls ]

train = pd.read_csv(sys.argv[1])
train_feature = train.feature.str.split(' ')
train_feature = train_feature.tolist()
private_pixels = np.array(train_feature, dtype=float)
private_pixels = private_pixels/255
private_pixels = private_pixels.reshape(private_pixels.shape[0],48,48,1)

choose_id = 17
photo = private_pixels[choose_id]
photo = photo.reshape(1,48,48,1)

for cnt, fn in enumerate(collect_layers):
    im = fn([photo, 0]) #get the output of that layer
    fig = plt.figure(figsize=(14, 8))
    nb_filter = 128
    for i in range(nb_filter):
        ax = fig.add_subplot(nb_filter/16, 16, i+1)
        ax.imshow(im[0][0, :, :, i], cmap='BuGn')
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.tight_layout()
    fig.suptitle('Output of layer{} (Given image{})'.format(cnt, choose_id))
    #fig.savefig(('./output_layer/layer'+str(cnt)))
    fig.savefig(('layer'+str(cnt)))

