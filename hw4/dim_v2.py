import numpy as np
import sys
from sklearn.neighbors import NearestNeighbors

data = np.load(sys.argv[1])
pred_dim = np.array([0.0]*200)
table = np.loadtxt('table.txt')

def classify(x):
    return (np.argmin(np.abs(table-x))+1)

for (k, i) in enumerate (data.keys()):
    temp_data = data[i]
    temp_data = temp_data[:5000]
    nbrs = NearestNeighbors(n_neighbors=2).fit(temp_data)
    distances, indices = nbrs.kneighbors(temp_data)
    std = distances[:,1].std()
    pred_dim[k] = np.log(classify(std))
    print ('Progress: ',k)


of = open(sys.argv[2],'w')
out = 'SetId,LogDim\n'
for k in range(len(pred_dim)):
    out = out + str(k) + ',' + str(pred_dim[k]) + '\n'

of.write(out)
of.close()
