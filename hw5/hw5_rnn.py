import numpy as np
import pickle
from keras.models import Sequential, load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.layers import Dense, Embedding, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers import LSTM,SimpleRNN,GRU
import keras.backend as K
from keras.optimizers import SGD
from keras.callbacks import CSVLogger
from keras import regularizers
from keras.constraints import maxnorm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import matthews_corrcoef, f1_score



embedding_dim = 100
val = 400 # the number of validation data
batch_size = 128
epoch_num =1
train_path = 'train_data.csv' #sys.argv[1]
test_path = 'test_data.csv'   #sys.argv[2]
#output_path = 'ans.csv'       #sys.argv[3]

def f1_measure(y_true,y_pred):
    thresh = 0.4
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred)
    
    precision=tp/(K.sum(y_pred))
    recall=tp/(K.sum(y_true))
    return 2*((precision*recall)/(precision+recall))


def get_embedding_dict(path):
    embedding_dict = {}
    with open(path,'r') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:],dtype='float32')
            embedding_dict[word] = coefs
    return embedding_dict


def get_embedding_matrix(word_index,embedding_dict,num_words,embedding_dim):
    embedding_matrix = np.zeros((num_words,embedding_dim))
    for word, i in word_index.items():
        if i < num_words:
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix


def read_data(path,train):
    f = open(path,'r').readlines()
    label = []
    txt = []
    f = f[1:]    #get rid of header

    if (train):
        print ('Parse the training data')
        for i in range(len(f)):
            d1 = f[i].find('"')
            d2 = f[i].find('"',d1+1)
            temp = f[i][d1:d2+1].strip().replace('"','').split(' ')
            label.append(temp)
            txt.append(f[i][d2+2:].strip())    # to prevent ',' at the begining
    else:
        print ('Parse the testing data')
        for i in range(len(f)):
            d1 = f[i].find(',')
            txt.append(f[i][d1+1:].strip())    # to prevent ',' at the begining

    return (txt,label) 


(txt_train,label) = read_data('train_data.csv',train=True)
(txt_test,_) = read_data('test_data.csv',train=False)

######### prepocess
print ('Convert to index sequences.')
corpus = txt_train + txt_test
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)

word_index = tokenizer.word_index

X_data = txt_train[:-val]
X_test = txt_train[-val:]
mlb = pickle.load(open('mlb.pkl','rb'))
label_num = len(mlb.classes_)
y_train = mlb.fit_transform(label)

y_test = y_train[-val:]
y_train = y_train[:-val]

train_sequences = tokenizer.texts_to_sequences(X_data)
test_sequences = tokenizer.texts_to_sequences(X_test)

pickle.dump(tokenizer,open('tokenizer.pkl','wb'))
print('Pad sequences:')
x_train = sequence.pad_sequences(train_sequences)
max_article_length = x_train.shape[1]
x_test = sequence.pad_sequences(test_sequences,maxlen=max_article_length)
#########

print ('Get embedding dict from glove.')
embedding_dict = get_embedding_dict('./glove.6B/glove.6B.%dd.txt'%embedding_dim)
print ('Found %s word vectors.' % len(embedding_dict))
max_features = len(word_index) + 1  # i.e. the number of words
print ('Create embedding matrix.')
embedding_matrix = get_embedding_matrix(word_index,embedding_dict,max_features,embedding_dim)

print('Build model...')

csv_logger = CSVLogger('training.csv')
model = Sequential()
model.add(Embedding(max_features,
                    embedding_dim,
                    weights=[embedding_matrix],
                    input_length=max_article_length,
                    trainable=False,
                   )
         )
model.add(LSTM(128, dropout=0.3,activation='tanh', recurrent_dropout=0.3))
model.add(Dense(label_num, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=[f1_measure])
model.summary()

model.fit(x_train, y_train,
      batch_size=batch_size,
      epochs=epoch_num,
      validation_data =(x_test, y_test),
      callbacks=[csv_logger]
     )

threshold = np.arange(0.1,0.9,0.1)
out = model.predict(x_test)

acc = []
accuracies = []
best_threshold = np.zeros(out.shape[1])
for i in range(out.shape[1]):
    y_prob = np.array(out[:,i])
    for j in threshold:
        y_pred = [1 if prob>=j else 0 for prob in y_prob]
        acc.append( matthews_corrcoef(y_test[:,i],y_pred))
    acc   = np.array(acc)
    index = np.where(acc==acc.max()) 
    accuracies.append(acc.max()) 
    best_threshold[i] = threshold[index[0][0]]
    acc = []

y_pred = np.array([[1 if out[i,j]>=best_threshold[j] else 0 for j in range(y_test.shape[1])] for i in range(len(y_test))])
of = open('threshold.txt','w')
out_txt = ''
for i in range(len(best_threshold)):
    out_txt += str(best_threshold[i]) + ' '

of.write(out_txt)
of.close()
print(f1_score(y_test, y_pred, average='micro'))

score = model.evaluate(x_train,y_train)

print ('\nTrain Acc: ', score[1])
model.save('my_model.h5')
