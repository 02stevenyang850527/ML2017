import pandas as pd
import numpy as np

train_path = 'train.csv'
test_path = 'test.csv'
output_path = 'ans.csv'

def matrix_factorization(R, P, Q, K, steps=701, lr=0.00001,beta=0.02):
    bias = np.zeros(R.shape)
    Q = Q.T
    for _ in range(steps):
        indice = np.where(R == 0)
        error = np.dot(P,Q) + bias -R
        error[indice] = 0
        eva = np.sqrt((error**2).sum()/(R!=0).sum())
        if (_%10 == 0):
            print ('iteration:{}, loss:{}'.format(_,eva))
        p_grad = 2*np.dot(error,Q.T)
        q_grad = 2*np.dot(P.T,error)
        b_grad = 2*error
        P -= lr*(p_grad-beta*P)
        Q -= lr*(q_grad-beta*Q)
        bias -= lr*(b_grad)
        if (eva < 0.85):
            print ('itrration break:',_)
            break;

    return (P,Q.T)


ratings = pd.read_csv(train_path)
test = pd.read_csv(test_path)

n_movie = np.max(ratings['MovieID'])
n_users = np.max(ratings['UserID'])
y = np.zeros((n_users,n_movie))

row = ratings['UserID'].values-1
col = ratings['MovieID'].values-1
y[row,col] = ratings['Rating'].values

np.random.seed(5)
(M,N) = y.shape
K = 5
print(M,N)

P = np.random.rand(M,K)
Q = np.random.rand(N,K)
nP, nQ = matrix_factorization(y, P, Q, K)

pred = np.dot(nP,nQ.T)

query = (test['UserID'].values-1,test['MovieID'].values-1)
result = pred[query]
of = open(output_path,'w')
out_txt = 'TestDataID,Rating\n'
for i in range(len(result)):
    out_txt += str(i+1) +','+ str(result[i]) + '\n'

of.write(out_txt)
of.close()
