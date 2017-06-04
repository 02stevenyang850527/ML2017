import pandas as pd
import numpy as np
from keras.models import load_model
import keras.backend as K


def rmse(y_true,y_pred):
    return K.sqrt(K.mean((y_pred - y_true)**2))


test_path = 'test.csv'
output_path = 'ans_nn.csv'

test = pd.read_csv(test_path)

user = test['UserID'].values
movie = test['MovieID'].values

model = load_model('mf.h5', custom_objects={'rmse':rmse})
result = model.predict([user, movie], batch_size=128, verbose=1)
result = result.reshape(len(result))

of = open(output_path,'w')
out_txt = 'TestDataID,Rating\n'
for i in range(len(result)):
    out_txt += str(i+1) +','+ str(result[i]) + '\n'

of.write(out_txt)
of.close()
