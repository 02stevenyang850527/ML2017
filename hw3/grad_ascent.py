import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
import numpy as np
NUM_STEPS = 20
RECORD_FREQ = 5

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-7)

def grad_ascent(num_step,input_image_data,iter_func,filter_images):
    """
    Implement this function!
    """
    lr = 1
    for i in range(num_step):
        loss, grad = iter_func([input_image_data,0])
        input_image_data += grad*lr
        if (i%RECORD_FREQ == 0):
            filter_images[int(i/RECORD_FREQ)].append([np.copy(np.reshape(input_image_data,(48,48))), loss])
    return filter_images

def main():
    emotion_classifier = load_model('my_model.h5')
    layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[1:])
    input_img = emotion_classifier.input
    num_steps = NUM_STEPS
    nb_filter = 112

    name_ls = ['activation_2']
    collect_layers = [ layer_dict[name].output for name in name_ls ]

    for cnt, c in enumerate(collect_layers):
        filter_imgs = [[] for i in range(NUM_STEPS//RECORD_FREQ)]
        for filter_idx in range(nb_filter):
            input_img_data = np.random.random((1, 48, 48, 1)) # random noise
            target = K.mean(c[:, :, :, filter_idx])
            grads = normalize(K.gradients(target, input_img)[0])
            iterate = K.function([input_img,K.learning_phase()], [target, grads])

            ###
            "You need to implement it."
            grad_ascent(num_steps, input_img_data, iterate, filter_imgs)
            ###

        for it in range(NUM_STEPS//RECORD_FREQ):
            fig = plt.figure(figsize=(14, 8))
            for i in range(nb_filter):
                ax = fig.add_subplot(nb_filter/16, 16, i+1)
                ax.imshow(filter_imgs[it][i][0], cmap='BuGn')
                plt.xticks(np.array([]))
                plt.yticks(np.array([]))
                plt.xlabel('{:.3f}'.format(filter_imgs[it][i][1]))
                plt.tight_layout()
            fig.suptitle('Filters of layer {} (# Ascent Epoch {} )'.format(name_ls[cnt], it*RECORD_FREQ))
            fig.savefig('e'+ str(it) + '_act2')

if __name__ == "__main__":
    main()
