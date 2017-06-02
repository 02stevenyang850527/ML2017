import pandas as pd
from numpy.linalg import inv
import numpy as np

train_path = 'train.csv'

def matrix_factorization(R, P, Q, K, Bu, Bi, steps=701, lr=0.1,beta=0.02):
    alpha_p = 0
    alpha_q = 0
    alpha_bu = np.zeros((1,R.shape[1]))
    alpha_bi = np.zeros((R.shape[0],1))
    Q = Q.T
    for _ in range(steps):
        indice = np.where(R == 0)
        #error = np.dot(P,Q) + bias -R
        error = (np.dot(P,Q) + Bu + Bi) - R
        error[indice] = 0
        eva = np.sqrt((error**2).sum()/(R!=0).sum())
        if (_%10 == 0):
            print ('iteration:{}, loss:{}'.format(_,eva))
        p_grad = 2*np.dot(error,Q.T)
        q_grad = 2*np.dot(P.T,error)
        alpha_p += np.diag(np.dot(p_grad,p_grad.T)).mean()
        alpha_q += np.diag(np.dot(q_grad.T,q_grad)).mean()
        alpha_bu += (2*error**2).mean(0)
        alpha_bi += (2*error**2).mean(1).reshape(-1,1)
        P -= lr/np.sqrt(alpha_p)*(p_grad-beta*P)
        Q -= lr/np.sqrt(alpha_q)*(q_grad-beta*Q)
        Bu -= lr/np.sqrt(alpha_bu.mean())*(2*error.mean(0)+beta*Bu)
        Bi -= lr/np.sqrt(alpha_bi.mean())*(2*error.mean(1).reshape(-1,1)+beta*Bi)
        if (eva < 0.075):
            print ('itrration break:',_)
            break;
    return (P,Q.T,Bu,Bi)


def svdpp(R, P, Q, K, Bu, Bi, y, steps=701, lr=0.1,beta=0.02):   # too long, abort
    alpha_p = 0
    alpha_q = 0
    u = R.shape[0]
    alpha_y = np.zeros((u,1))
    alpha_bu = np.zeros((1,R.shape[1]))
    alpha_bi = np.zeros((R.shape[0],1))
    Q = Q.T
    indice = (R == 0)
    for _ in range(steps):
        temp = np.array([(indice[i]*y[i].T).sum(1)/indice[i].sum() for i in range(u)])
        error = np.dot(P + temp,Q) + Bu + Bi - R
        error[indice] = 0
        eva = np.sqrt((error**2).sum()/(R!=0).sum())
        #if (_%10 == 0):
        print ('iteration:{}, loss:{}'.format(_,eva))
        p_grad = 2*np.dot(error,Q.T)
        q_grad = 2*np.dot((P + temp).T,error)
        y_grad = np.array([(error[i]*Q).T for i in range(len(error))])  #y_grad.shape = M*N*k

        alpha_p += np.diag(np.dot(p_grad,p_grad.T)).mean()
        alpha_q += np.diag(np.dot(q_grad.T,q_grad)).mean()
        alpha_y += np.array([np.diag(np.dot(y_grad[i],y_grad[i].T)).mean() for i in range(u)])
        alpha_bu += (2*error**2).mean(0)
        alpha_bi += (2*error**2).mean(1).reshape(-1,1)

        P -= lr/np.sqrt(alpha_p)*(p_grad-beta*P)
        Q -= lr/np.sqrt(alpha_q)*(q_grad-beta*Q)
        Bu -= lr/np.sqrt(alpha_bu.mean())*(2*error.mean(0)-beta*Bu)
        Bi -= lr/np.sqrt(alpha_bi.mean())*(2*error.mean(1).reshape(-1,1)-beta*Bi)
        y -= np.array([lr/np.sqrt(alpha_y[i])*(y_grad[i]-beta*y[i]) for i in range(u)])
        if (eva < 0.75):
            print ('itrration break:',_)
            break;
    return (P,Q.T,Bu,Bi)


ratings = pd.read_csv(train_path)

n_movie = np.max(ratings['MovieID'])
n_users = np.max(ratings['UserID'])
y = np.zeros((n_users,n_movie))

row = ratings['UserID'].values-1
col = ratings['MovieID'].values-1
rat = ratings['Rating'].values
mean = rat.mean()
std  = rat.std()
#rat = (rat-mean)/std
y[row,col] = rat
print(mean,std)

np.random.seed(5)
(M,N) = y.shape
K = 10
print(M,N)

P = np.random.rand(M,K)
Q = np.random.rand(N,K)
Bu = np.random.rand(1,N) # user's preference
Bi = np.random.rand(M,1) # item's attraction

#yy = np.zeros((M,N,K)) # latent y
#nP, nQ, nBu, nBi= svdpp(y, P, Q, K, Bu,Bi,yy)
nP, nQ, nBu, nBi= matrix_factorization(y, P, Q, K, Bu,Bi)

pred = np.dot(nP,nQ.T) + nBu + nBi
#pred = pred*std + mean
np.save('pred',pred)

