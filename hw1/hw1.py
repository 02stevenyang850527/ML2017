import pandas as pd
import numpy as np

#----------------- read train.csv -------------------
data = pd.read_csv('train.csv', encoding = 'big5')
data[data == "NR"] = 0
data_filter = data.iloc[:,2:]
data_str = np.array(data_filter)

#---------------- read test_X.csv -------------------
data_t = pd.read_csv('test_X.csv',header=None)
data_t[data_t == 'NR'] = 0
data_t = np.array(data_t)
#---------------- extract training data -------------
PM25 = []
data_list = data_str.tolist()

for i in range(12):
    new = []
    for j in range(18):
        '''if (data_list[j][0] == 'AMB_TEMP'):
            for k in range(20):
                new = new + data_list[i*20*18+k*18+j][1:]
        elif (data_list[j][0] == 'CH4'):
            for k in range(20):
                new = new + data_list[i*20*18+k*18+j][1:]
        elif (data_list[j][0] == 'CO'):
            for k in range(20):
                new = new + data_list[i*20*18+k*18+j][1:]
        elif (data_list[j][0] == 'NMHC'):
            for k in range(20):
                new = new + data_list[i*20*18+k*18+j][1:]
        elif (data_list[j][0] == 'NO'):
            for k in range(20):
                new = new + data_list[i*20*18+k*18+j][1:]
        elif (data_list[j][0] == 'NO2'):
            for k in range(20):
                new = new + data_list[i*20*18+k*18+j][1:]
        elif (data_list[j][0] == 'NOx'):
            for k in range(20):
                new = new + data_list[i*20*18+k*18+j][1:]'''
        if (data_list[j][0] == 'O3'):
            for k in range(20):
                new = new + data_list[i*20*18+k*18+j][1:]
        elif (data_list[j][0] == 'PM10'):
            for k in range(20):
                new = new + data_list[i*20*18+k*18+j][1:]
        elif (data_list[j][0] == 'PM2.5'):
            for k in range(20):
                new = new + data_list[i*20*18+k*18+j][1:]
        elif (data_list[j][0] == 'RAINFALL'):
            for k in range(20):
                new = new + data_list[i*20*18+k*18+j][1:]
        elif (data_list[j][0] == 'WD_HR'):
            for k in range(20):
                new = new + data_list[i*20*18+k*18+j][1:]
        elif (data_list[j][0] == 'WIND_DIREC'):
            for k in range(20):
                new = new + data_list[i*20*18+k*18+j][1:]
        elif (data_list[j][0] == 'WIND_SPEED'):
            for k in range(20):
                new = new + data_list[i*20*18+k*18+j][1:]
        elif (data_list[j][0] == 'WS_HR'):
            for k in range(20):
                new = new + data_list[i*20*18+k*18+j][1:]
        '''elif (data_list[j][0] == 'RH'):
            for k in range(20):
                new = new + data_list[i*20*18+k*18+j][1:]
        elif (data_list[j][0] == 'SO2'):
            for k in range(20):
                new = new + data_list[i*20*18+k*18+j][1:]
        elif (data_list[j][0] == 'THC'):
            for k in range(20):
                new = new + data_list[i*20*18+k*18+j][1:]'''
    PM25.append(new)

feature_num = 8
PM25 = np.array(PM25)
PM25 = PM25.astype(np.float)
#print (PM25[0,480:960])
train_data = []
train_result = []
for i in range(12):
    temp2 = PM25[i].reshape((feature_num,-1))
    #print (temp2[1])
    for j in range(471):
        temp1 = np.array([])
        for k in range(feature_num):
            temp1 = np.concatenate((temp1,temp2[k][j:j+9]))
        train_result.append(temp2[2][j+9])
        train_data.append(temp1)

train_data = np.array(train_data)
train_result = np.array(train_result)
train_data = np.concatenate((train_data, train_data[:, 9:27]**2),axis=1)
train_data = np.concatenate((train_data, train_data[:,18:27]**3), axis=1)

#train_data = train_data[450:,:]
#train_result = train_result[450:]
mean = train_data.mean(0)
std = train_data.std(0)
train_data = (train_data-mean)/std

#------------------- training ---------------------
b = 0.0
w_pm25 = np.array([0.0]*(train_data.shape[1]))
lr = 0.5
b_lr = 0.0
w_lr_pm25 = np.array([0.0]*(train_data.shape[1]))
iteration = 15000
for i in range(iteration):
   # b_grad = 0.0;
   # w_grad_pm25 = np.array([0.0]*11)
    scalar_ma = train_result - b - np.dot(train_data,w_pm25.T) # 1*n matrix
    b_grad = -2.0*np.sum(scalar_ma)
    w_grad_pm25 = -2.0*np.dot(scalar_ma,train_data)
    b_lr = b_lr + b_grad**2
    w_lr_pm25 = w_lr_pm25 + w_grad_pm25**2
    # Update parameters
    b = b - lr/np.sqrt(b_lr)*b_grad
    w_pm25 = w_pm25 - lr/np.sqrt(w_lr_pm25)*w_grad_pm25
    if (i%100 == 0):
        print ('iteration: {}, Loss = {}'.format(i, np.sqrt(np.mean((train_result-b-np.dot(train_data,w_pm25.T))**2))))

print(b)
print(w_pm25)

#------------------- test ------------------------
#test_feature = []
result = []
temp = np.array([])
for i in range(len(data_t)):
    '''if (data_t[i][1] == 'AMB_TEMP'):
        temp = np.concatenate((temp,data_t[i][2:].astype(np.float)))
    elif (data_t[i][1] == 'CH4'):
        temp = np.concatenate((temp,data_t[i][2:].astype(np.float)))
    elif (data_t[i][1] == 'CO'):
        temp = np.concatenate((temp,data_t[i][2:].astype(np.float)))
    elif (data_t[i][1] == 'NMHC'):
        temp = np.concatenate((temp,data_t[i][2:].astype(np.float)))
    elif (data_t[i][1] == 'NO'):
        temp = np.concatenate((temp,data_t[i][2:].astype(np.float)))
    elif (data_t[i][1] == 'NO2'):
        temp = np.concatenate((temp,data_t[i][2:].astype(np.float)))
    elif (data_t[i][1] == 'NOx'):
        temp = np.concatenate((temp,data_t[i][2:].astype(np.float)))'''
    if (data_t[i][1] == 'O3'):
        temp = np.concatenate((temp,data_t[i][2:].astype(np.float)))
    elif (data_t[i][1] == 'PM10'):
        temp = np.concatenate((temp,data_t[i][2:].astype(np.float)))
    elif (data_t[i][1] == 'PM2.5'):
        temp = np.concatenate((temp,data_t[i][2:].astype(np.float)))
    elif (data_t[i][1] == 'RAINFALL'):
        temp = np.concatenate((temp,data_t[i][2:].astype(np.float)))
    elif (data_t[i][1] == 'WD_HR'):
        temp = np.concatenate((temp,data_t[i][2:].astype(np.float)))
    elif (data_t[i][1] == 'WIND_DIREC'):
        temp = np.concatenate((temp,data_t[i][2:].astype(np.float)))
    elif (data_t[i][1] == 'WIND_SPEED'):
        temp = np.concatenate((temp,data_t[i][2:].astype(np.float)))
    elif (data_t[i][1] == 'WS_HR'):
        temp = np.concatenate((temp,data_t[i][2:].astype(np.float)))
    '''elif (data_t[i][1] == 'RH'):
        temp = np.concatenate((temp,data_t[i][2:].astype(np.float)))
    elif (data_t[i][1] == 'SO2'):
        temp = np.concatenate((temp,data_t[i][2:].astype(np.float)))
    elif (data_t[i][1] == 'THC'):
        temp = np.concatenate((temp,data_t[i][2:].astype(np.float)))'''
        #temp = np.append(temp,temp**2)
    if (i%18 == 17):
        temp = np.concatenate((temp,temp[9:27]**2))
        temp = np.concatenate((temp,temp[18:27]**3))
        result.append(b + np.dot(w_pm25,(temp-mean)/std))
        temp = np.array([])

print (result)
of = open('submit.csv','w')
out = 'id,value'
for i in range(len(result)):
    out = out + '\n' + 'id_' + str(i) +',' + str(result[i])
of.write(out+'\n')
of.close()

