import pandas as pd
import numpy as np

def sigmoid(x):
    try:
        return 1/(1+np.exp(0.0-x))
    except OverflowError:
        return 1.0

#-------read data--------
feature = pd.read_csv('X_train')
raw_data = pd.read_csv('train.csv',header=None)
feature_add = np.array(raw_data.iloc[:,4])
feature_add = np.concatenate((feature_add.reshape(-1,1)**2,(feature_add.reshape(-1,1)**3)),axis=1)
feature_np = np.array(feature)
feature_np = np.concatenate((feature_np,feature_add),axis=1)
feature_np = np.concatenate((feature_np,feature_np[:,:6]**2),axis=1)
feature_np = np.concatenate((feature_np,feature_np[:,:6]**3),axis=1)
feature_mul0 = (feature_np[:,0]**3)*(feature_np[:,2]**3) #sex
feature_mul1 = (feature_np[:,0]**3)*(feature_np[:,3]**3) #capital_gain
feature_np = np.concatenate((feature_np,feature_mul0.reshape(-1,1)),axis=1)
feature_np = np.concatenate((feature_np,feature_mul1.reshape(-1,1)),axis=1)
mean = feature_np.mean(0)
std = feature_np.std(0)
feature_np = (feature_np-mean)/std

train_result = np.loadtxt('Y_train')
test_data = pd.read_csv('X_test')
test_raw = pd.read_csv('test.csv')
test_add = np.array(test_raw.iloc[:,4])
test_add = np.concatenate((test_add.reshape(-1,1)**2,(test_add.reshape(-1,1))**3),axis=1)
test_np = np.array(test_data)
test_np = np.concatenate((test_np,test_add),axis=1)
test_np = np.concatenate((test_np, test_np[:,:6]**2),axis=1)
test_np = np.concatenate((test_np, test_np[:,:6]**3),axis=1)
test_mul0 = (test_np[:,0]**3)*(test_np[:,2]**3)
test_mul1 = (test_np[:,0]**3)*(test_np[:,3]**3)
test_np = np.concatenate((test_np,test_mul0.reshape(-1,1)),axis=1)
test_np = np.concatenate((test_np,test_mul1.reshape(-1,1)),axis=1)
test_np = (test_np-mean)/std
#------- train ----------

b = 0.0
w = np.array([0.0]*feature_np.shape[1])
lr = 0.5
b_lr = 0.0
w_lr = np.array([0.0]*feature_np.shape[1])
lamda = 0  # For Regularization
iteration = 2000

for i in range(iteration):
    scalar_mat = train_result - sigmoid(b + np.dot(feature_np,w))
    b_grad = -np.sum(scalar_mat)
    w_grad = -np.dot(scalar_mat,feature_np) + lamda*w
    b_lr = b_lr + b_grad**2
    w_lr = w_lr + w_grad**2
    # Update parameters
    b = b - lr/np.sqrt(b_lr)*b_grad
    w = w - lr/np.sqrt(w_lr)*w_grad
    # print cross entropy
    if (i%100 == 1):
        z = np.dot(feature_np,w)+b
        y = sigmoid(z)
        cross_entropy = -(np.dot(train_result,np.log(y)) + np.dot(1-train_result,np.log(1-y)))
        print ('iteration: {}, cross_entropy = {}'.format(i,cross_entropy))

#--------- test ----------
counter = 0
p_train = sigmoid(np.dot(feature_np,w)+b)
for i in range(len(feature_np)):
    if (p_train[i] > 0.5):
        if (train_result[i]==1):
            counter = counter+1
    else:
        if (train_result[i]==0):
            counter = counter+1

print ('accuracy on training: ', counter/len(feature_np))

prediction = np.array([0]*len(test_np))
p = sigmoid(np.dot(test_np,w)+b)

for j in range(len(test_np)):
    if (p[j] > 0.5):
        prediction[j] = 1
    else:
        prediction[j] = 0

of = open('prediction.csv','w')
out = 'id,label\n'
for k in range(len(prediction)):
    out = out + str(k+1) + ',' + str(prediction[k]) + '\n'

of.write(out)
of.close()


