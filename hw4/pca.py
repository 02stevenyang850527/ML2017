import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

cla = ['A','B','C','D','E','F','G','H','I','J']
num = 10
pic = []

for c in cla:
    for i in range(num):
        pic.append(misc.imread('./faceExpressionDatabase/'+c+
                    '0'+str(i)+'.bmp').reshape(1,-1).astype('float64'))

pic = np.array(pic)
pic = pic.reshape(-1,64*64)

def est_error(e_face):
    return np.sqrt(((e_face-pic)**2).sum()/(e_face.shape[0]*e_face.shape[1]))/256

fig0 = plt.figure()
pic_mean = pic.mean(axis=0, keepdims=True)
plt.imshow(pic_mean.reshape(64,64),cmap='gray')
plt.tight_layout()
plt.colorbar()
plt.title('Average Face')
fig0.savefig('./pca/ave_face.png')


pic_ctr = pic - pic_mean
U, s, V = np.linalg.svd(pic_ctr, full_matrices=False)

sel = 9
eig_face = V[:sel,:]

fig1 = plt.figure(figsize=(12,6))
for i in range(sel):
    ax = fig1.add_subplot(sel/3, 3, i+1)
    ax.imshow(eig_face[i,:].reshape(64,64), cmap='gray')
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.xlabel('eigenface_rank#{}'.format(i+1))
    plt.tight_layout()
fig1.suptitle('Top 9 eigenface')
fig1.savefig('./pca/top_9_eigenface.png')


k = 5
weight = np.dot(pic_ctr,V[:k,:].transpose())
est_face = np.dot(weight,V[:k,:])
fig2 = plt.figure(figsize=(12,9))
for i in range(est_face.shape[0]):
    ax = fig2.add_subplot(est_face.shape[0]/10,10,i+1)
    ax.imshow(est_face[i,:].reshape(64,64)+pic_mean.reshape(64,64), cmap='gray')
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.xlabel('id_{}'.format(i+1))
    plt.tight_layout()
fig2.suptitle('Recovered face after PCA')
fig2.savefig('./pca/recovered_face_100.png')


fig3 = plt.figure(figsize=(12,9))
for i in range(pic.shape[0]):
    ax = fig3.add_subplot(pic.shape[0]/10,10,i+1)
    ax.imshow(pic[i,:].reshape(64,64), cmap='gray')
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.xlabel('id_{}'.format(i+1))
    plt.tight_layout()
fig3.suptitle('Original face')
fig3.savefig('./pca/original_face_100.png')


k_candi = list(range(100))
for i in k_candi:
    w = np.dot(pic_ctr,V[:i,:].transpose())
    e_face = np.dot(w,V[:i,:]) + pic_mean
    err = est_error(e_face)
    print('k: {}; err: {}'.format(i,err))
    if (err < 0.01):
        print('Error below 0.01, and k is ',i)
        break
