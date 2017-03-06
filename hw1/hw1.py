import pandas as pd, numpy as np

#----------------- read train.csv -------------------
data = pd.read_csv('train.csv', encoding = 'big5')
data[data == "NR"] = 0
data_filter = data.iloc[:,2:]
data_str = np.array(data_filter)

#---------------- extract training data -------------
PM25_feature = [];
result = [];

for i in range(12):
    






for i in range(4320):
    temp_f = [] #temp for features
    temp_r = [] #temp for results
    if (data_str[i][0] == "PM2.5"):
        for j in range(15):
            temp.append(data_str[i][j+1:j+9])
            temp_r.append(float(data_str[i][j+10]))
            result.append(temp_r)
            PM25_feature.append(temp)

# print (PM25_feature[0]) --> one day data, total = 16
# print (PM25_feature[0][0]) --> one training data
# print (PM25_feature[0][0][0]) --> one hour data

#--------------- gradient descent ------------------
b = 10
w = 4



