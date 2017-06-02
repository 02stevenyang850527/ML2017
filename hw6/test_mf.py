import numpy as np
import pandas as pd


test_path = 'test.csv'
output_path = 'ans_mf.csv'

test = pd.read_csv(test_path)
pred = np.load('pred.npy')
query = (test['UserID'].values-1,test['MovieID'].values-1)
result = pred[query]

out_txt = 'TestDataID,Rating\n'
of = open(output_path,'w')
for i in range(len(result)):
    out_txt += str(i+1) +','+ str(result[i]) + '\n'

of.write(out_txt)
of.close()
