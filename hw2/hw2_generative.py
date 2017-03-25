import pandas as pd
import numpy as np

#-------read data--------
feature = pd.read_csv('X_train')
#feature_selected
feature_np = np.array(frature)
#raw_data = pd.read_csv('train.csv',header=None)
#data = np.array(raw_data)
train_result = np.loadtxt('Y_train')

#-------classification-----
model_0 = feature_np[(train_result==1),:]
model_1 = feature_np[(train_result==1),:]
number_0 = model_0.shape[0]
number_1 = model_0.shape[1]
p0 = number_0/(number_0+number_1)
p1 = number_1/(number_0+number_1)
u0 = model_0.mean(0)
u1 = model_1.mean(0)
sigma0 = (np.cov(model_0.T))/number_0
sugma1 = (np.cov(model_1.T))/number_1


