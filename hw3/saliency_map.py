from keras.models import load_model
from termcolor import colored,cprint
import keras.backend as K
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

np.seterr(divide='ignore', invalid='ignore')
train = pd.read_csv(sys.argv[1])
train_feature = train.feature.str.split(' ')
train_feature = train_feature.tolist()
train_feature = np.array(train_feature, dtype=float)
train_label = np.array(train['label'])
train_feature = train_feature.reshape(train_feature.shape[0],48,48,1)

model = load_model('my_model.h5')
input_img = model.input
img_ids = list(range(10))

for idx in img_ids:
    origin = train_feature[idx].reshape(48, 48)
    plt.figure()
    plt.imshow(origin,cmap='gray')
    plt.colorbar()
    plt.tight_layout()
    fig = plt.gcf()
    plt.draw()
    #fig.savefig('./pic/'+str(idx)+".png", dpi=100)
    fig.savefig(str(idx)+".png", dpi=100)

    val_proba = model.predict((train_feature[idx]/255).reshape(1,48,48,1))
    pred = val_proba.argmax(axis=-1)
    target = K.mean(model.output[:, pred])
    grads = K.gradients(target, input_img)[0]
    fn = K.function([input_img, K.learning_phase()], [grads])

    g = fn([train_feature[idx].reshape(1,48,48,1)/255,0])
    g = g[0]
    g -= g.mean()
    g /= (g.std()+1e-20)
    g *= 0.1
    g += 0.5
    g = np.clip(g,0,1)
    heatmap = g.reshape(48,48)
    #heatmap = np.uint8(train_feature[idx].reshape(48,48)*0.5 + heatmap*(1. - 0.5))
    '''
    Implement your heatmap processing here
    hint: Do some normalization or smoothening on grads
    '''
    
    thres = 0.55
    see = train_feature[idx].reshape(48, 48)
    loc = heatmap <= thres
    loc = loc.reshape(48,48)
    heatmap = heatmap.reshape(48,48)
    see[loc] = np.mean(see)
    
    plt.figure()
    plt.imshow(heatmap, cmap=plt.cm.jet)
    plt.colorbar()
    plt.tight_layout()
    fig = plt.gcf()
    plt.draw()
    #fig.savefig("./pic/"+str(idx)+"_heatmap.png",dpi=100)
    fig.savefig(str(idx)+"_heatmap.png",dpi=100)
    
    plt.figure()
    plt.imshow(see,cmap='gray')
    plt.colorbar()
    plt.tight_layout()
    fig = plt.gcf()
    plt.draw()
    #fig.savefig('./pic/'+str(idx)+"_mask.png", dpi=100)
    fig.savefig(str(idx)+"_mask.png", dpi=100)
    
