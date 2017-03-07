import pandas as pd, numpy as np

#----------------- read train.csv -------------------
data = pd.read_csv('train.csv', encoding = 'big5')
data[data == "NR"] = 0
data_filter = data.iloc[:,2:]
data_str = np.array(data_filter)

#---------------- read test_X.csv -------------------
data_t = pd.read_csv('test_X.csv')
data_t = np.array(data_t)

#---------------- extract training data -------------
PM25 = []
data_list = data_str.tolist()

for i in range(12):
    new = []
    for j in range(360):
        if (data_list[i*360 + j][0] == 'PM2.5'):
             new = new + data_list[i*360 + j][1:]
    PM25.append(new)

PM25 = np.array(PM25)
PM25 = PM25.astype(np.float)

#------------------- training ---------------------
b = 0
w_pm25 = np.array([0.0]*9)
lr = 0.5
b_lr = 0.0
w_lr_pm25 = np.array([0.0]*9)
iteration = 400

for i in range(iteration):
    b_grad = 0.0;
    w_grad_pm25 = np.array([0.0]*9)
    for m in range(len(PM25)): # 12 month
        for n in range(len(PM25[0])-9): # (480-9) data
            b_grad = b_grad - 2.0*(PM25[m][n+9]- b -sum(w_pm25*(PM25[m][n:n+9])))
            w_grad_pm25 = w_grad_pm25 - 2.0*(PM25[m][n+9] - b - sum(w_pm25*(PM25[m][n:n+9])))*PM25[m][n:n+9]
        b_lr = b_lr + b_grad**2
        w_lr_pm25 = w_lr_pm25 + w_grad_pm25**2
        # Update parameters
        b = b - lr/np.sqrt(b_lr)*b_grad
        w_pm25 = w_pm25 - lr/np.sqrt(w_lr_pm25)*w_grad_pm25

print(b)
print(w_pm25)
#------------------- test ------------------------
#test_feature = []
result = []
for i in range(len(data_t)):
    if (data_t[i][1] == 'PM2.5'):
        result.append(b + sum(w_pm25*(data_t[i][2:].astype(np.float))))
#print (result)
of = open('submit.csv','w')
out = 'id,value'
for i in range(len(result)):
    out = out + '\n' + 'id_' + str(i) +',' + str(result[i])
of.write(out+'\n')
of.close()
