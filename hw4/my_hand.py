import numpy as np
from scipy import misc
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVR as SVR
from sklearn.neighbors import NearestNeighbors


np.random.seed(35)
def get_eigenvalues(data):
    SAMPLE = 10 # sample some points to estimate
    NEIGHBOR = 200 # pick some neighbor to compute the eigenvalues
    randidx = np.random.permutation(data.shape[0])[:SAMPLE]
    knbrs = NearestNeighbors(n_neighbors=NEIGHBOR,
                             algorithm='ball_tree').fit(data)
    sing_vals = []
    for idx in randidx:
        dist, ind = knbrs.kneighbors(data[idx:idx+1])
        nbrs = data[ind[0,1:]]
        u, s, v = np.linalg.svd(nbrs - nbrs.mean(axis=0))
        s /= s.max()
        sing_vals.append(s)
    sing_vals = np.array(sing_vals).mean(axis=0)
    return sing_vals


pic = []
for i in range(1,482):
    pic.append(misc.imread('./hand/hand.seq'+str(i)+'.png').reshape(1,-1).astype('float64'))

pic = np.array(pic)
pic = pic.reshape(481,-1)

print ('PCA')
pca = PCA(n_components = 100)
testdata = pca.fit_transform(pic - pic.mean(axis=0))
# Train a linear SVR
npzfile = np.load('train_data.npz')
X = npzfile['X']
y = npzfile['y']

print ('Training')
svr = SVR(C=3)
svr.fit(X, y)


test_X = []
print('Process: ')
vs = get_eigenvalues(testdata)
test_X.append(vs)

test_X = np.array(test_X)
pred_y = svr.predict(test_X)

print('dim: ',pred_y)
