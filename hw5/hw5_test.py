import numpy as np
import sys 
import pickle
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.metrics import matthews_corrcoef
import keras.backend as K
from sklearn.externals import joblib

test_path = 'sys.argv[1]'    #test_data.csv
output_path = 'sys.argv[2]'

def f1_measure(y_true,y_pred):
    thresh = 0.4
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred,axis=-1)
    
    precision=tp/(K.sum(y_pred,axis=-1)+K.epsilon())
    recall=tp/(K.sum(y_true,axis=-1)+K.epsilon())
    return K.mean(2*((precision*recall)/(precision+recall+K.epsilon())))


model = load_model('my_model.h5',custom_objects={'f1_measure':f1_measure})
model_1 = load_model('my_model_1.h5',custom_objects={'f1_measure':f1_measure})
model_2 = load_model('my_model_2.h5',custom_objects={'f1_measure':f1_measure})
model_3 = load_model('my_model_3.h5',custom_objects={'f1_measure':f1_measure})
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
classif = joblib.load('model_sk.pkl')

mlb = pickle.load(open('mlb.pkl','rb'))
tokenizer = pickle.load(open('tokenizer.pkl','rb'))

word_index = tokenizer.word_index

f = open(test_path,'r').readlines()
txt = []
batch_size = 128
f = f[1:]

print ('Parse the data')
for i in range(len(f)):
    d1 = f[i].find(',')
    txt.append(f[i][d1+1:].strip())

test_sequences = tokenizer.texts_to_sequences(txt)
x_test_v =  vectorizer.transform(txt)

print('Pad sequences:')
x_test = sequence.pad_sequences(test_sequences,maxlen=306) # maxlen being known from hw5_rnn.py

print ('x_test shape:',x_test.shape)

out_0 = model.predict(x_test,batch_size=batch_size,verbose=1)
out_1 = model_1.predict(x_test,batch_size=batch_size,verbose=1)
out_2 = model_2.predict(x_test,batch_size=batch_size,verbose=1)
out_3 = model_3.predict(x_test,batch_size=batch_size,verbose=1)

y_pred_0 = np.array([[1 if out_0[i,j]>=0.4 else 0
                        for j in range(out_0.shape[1])]
                            for i in range(out_0.shape[0])]
                 )
y_pred_1 = np.array([[1 if out_1[i,j]>=0.4 else 0
                        for j in range(out_1.shape[1])]
                            for i in range(out_1.shape[0])]
                 )
y_pred_2 = np.array([[1 if out_2[i,j]>=0.4 else 0
                        for j in range(out_2.shape[1])]
                            for i in range(out_2.shape[0])]
                 )
y_pred_3 = np.array([[1 if out_3[i,j]>=0.4 else 0
                        for j in range(out_3.shape[1])]
                            for i in range(out_3.shape[0])]
                 )
y_pred_4 = classif.predict(x_test_v)

y_pred_s = y_pred_0 + y_pred_1 + y_pred_2 + y_pred_3 + y_pred_4

y_pred = np.array([[1 if y_pred_s[i,j]>=3 else 0
                        for j in range(y_pred_s.shape[1])]
                            for i in range(y_pred_s.shape[0])]
                 )

result = mlb.inverse_transform(y_pred)

of = open(output_path,'w')
out_txt = '"id","tags"\n'

for i in range(len(result)):
    out_txt += '"'+str(i)+'"'+',"'
    if (result[i]==()):
        out_txt += 'SPECULATIVE-FICTION' + ' '
    else:
        for te in result[i]:
            out_txt += te + ' '
    out_txt = out_txt[:-1]
    out_txt += '"\n'

of. write(out_txt)
of.close
