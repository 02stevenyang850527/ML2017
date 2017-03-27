import pandas as pd
import numpy as np

#-------read data--------
feature = pd.read_csv('X_train')
#feature_selected = feature[['age','fnlwgt','capital_gain','capital_loss','hours_per_week']]
feature_selected = feature
feature_np = np.array(feature_selected)
mean = feature_np.mean(0)
std = feature_np.std(0)
feature_np = (feature_np-mean)/std
train_result = np.loadtxt('Y_train')

test_data = pd.read_csv('X_test')
#test_selected = test_data[['age','fnlwgt','capital_gain','capital_loss','hours_per_week']]
test_selected = test_data
test_np = np.array(test_selected)
test_np = (test_np-mean)/std
#-------classification-----
model_0 = feature_np[(train_result==0),:]
model_1 = feature_np[(train_result==1),:]
number_0 = model_0.shape[0]
number_1 = model_1.shape[0]
p0 = number_0/(number_0+number_1)
p1 = number_1/(number_0+number_1)
u0 = model_0.mean(0)
u1 = model_1.mean(0)
sigma0 = (np.dot((model_0-u0).T,(model_0-u0)))/number_0
sigma1 = (np.dot((model_1-u1).T,(model_1-u1)))/number_1
#sigma = p0*sigma0 + p1*sigma1
det0 = np.linalg.det(sigma0)
det1 = np.linalg.det(sigma1)
inv_sigma0 = np.linalg.pinv(sigma0)
inv_sigma1 = np.linalg.pinv(sigma1)

#-------prediction---------
'''counter = 0
for i in range(len(train_result)):
    temp0 = feature_np[i]-u0
    temp1 = feature_np[i]-u1
    numerator = p1*np.exp(-0.5*np.dot(np.dot(temp1[None,:],inv_sigma),temp1[:,None]))
    denominator = p0*np.exp(-0.5*np.dot(np.dot(temp0[None,:],inv_sigma),temp0[:,None]))
    denominator += numerator
    if (numerator == 0):
        p = 0
    else:
        p = numerator/denominator

    if (p > 0.5):
        if (train_result[i]==1):
            counter = counter + 1
    else:
        if (train_result[i]==0):
            counter = counter + 1

print ('acc is ', counter/len(train_result))'''

counter = 0
prediction = np.array([0]*len(test_np))
for i in range(len(test_np)):
    temp0 = test_np[i]-u0
    temp1 = test_np[i]-u1
    numerator = p1*np.exp(-0.5*np.dot(np.dot(temp1[None,:],inv_sigma1),temp1[:,None]))
    denominator = p0*np.exp(-0.5*np.dot(np.dot(temp0[None,:],inv_sigma0),temp0[:,None]))
    denominator += numerator
    if (numerator == 0):
        p = 0
    else:
        p = numerator/denominator
    if (p > 0.5):
        prediction[i]=1
        counter = counter+1
    else:
        prediction[i]=0

print (counter)
of = open('prediction.csv','w')
out = 'id,label\n'
for k in range(len(prediction)):
    out = out + str(k+1) + ',' + str(prediction[k]) + '\n'

of.write(out)
of.close()
