import numpy as np
import sys

data = np.load(sys.argv[1])
thres = 2300

def condition(x):
    for i in range(len(x)):
        if x[i] < thres:
            return (i-1)
            break

pred_dim = np.array([0.0]*200)
for i in data.keys():
    temp_data = data[i].transpose()
    temp_data -= temp_data.mean(0)
    #U,s0,V = np.linalg.svd(temp_data,full_matrices=False)
    s = np.linalg.svd(temp_data,full_matrices=False,compute_uv=False)
    pred_dim[int(i)] = np.log(condition(s))
    print('Progress:'+i)

of = open(sys.argv[2],'w')
out = 'SetId,LogDim\n'
for k in range(len(pred_dim)):
    out = out + str(k) + ',' + str(pred_dim[k]) + '\n'

of.write(out)
of.close() 
