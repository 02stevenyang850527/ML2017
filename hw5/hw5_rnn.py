import numpy as np
import pickle
from keras.models import Sequential, load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.layers import Dense, Embedding, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv1D, MaxPooling1D
from keras.layers.wrappers import Bidirectional
from keras.layers import LSTM,SimpleRNN,GRU
import keras.backend as K
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD, Adam
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from keras import regularizers
from keras.constraints import maxnorm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import matthews_corrcoef, f1_score



embedding_dim = 200
split_ratio = 0.1 # the ratio of validation data
batch_size = 128
epoch_num = 55
train_path = 'train_data.csv' #sys.argv[1]
test_path = 'test_data.csv'   #sys.argv[2]
#output_path = 'ans.csv'       #sys.argv[3]

def f1_measure(y_true,y_pred):
    thresh = 0.4
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred,axis=-1)
    
    precision=tp/(K.sum(y_pred,axis=-1)+K.epsilon())
    recall=tp/(K.sum(y_true,axis=-1)+K.epsilon())
    return K.mean(2*((precision*recall)/(precision+recall+K.epsilon())))


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


def split_data(X,Y,split_ratio):
    np.random.seed(5)
    indices = np.random.permutation(X.shape[0])
    
    X_data = X[indices]
    Y_data = Y[indices]
    
    num_validation_sample = int(split_ratio * X_data.shape[0] )
    
    X_train = X_data[num_validation_sample:]
    Y_train = Y_data[num_validation_sample:]

    X_val = X_data[:num_validation_sample]
    Y_val = Y_data[:num_validation_sample]

    return (X_train,Y_train),(X_val,Y_val)


def main():
    (txt_train,label) = read_data(train_path,train=True)
    (txt_test,_) = read_data(test_path,train=False)

    ######### prepocess
    print ('Convert to index sequences.')
    corpus = txt_train + txt_test
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)

    word_index = tokenizer.word_index

    mlb = pickle.load(open('mlb.pkl','rb'))
    label_num = len(mlb.classes_)
    Y = mlb.fit_transform(label)

    train_sequences = tokenizer.texts_to_sequences(txt_train)

    pickle.dump(tokenizer,open('tokenizer.pkl','wb'))
    print('Pad sequences:')
    X = sequence.pad_sequences(train_sequences)
    print('Split data into training data and validation data')
    (x_train,y_train),(x_test,y_test) = split_data(X,Y,split_ratio)
    max_article_length = x_train.shape[1]
    #########
    print ('maxlen',max_article_length)
    print ('Get embedding dict from glove.')
    embedding_dict = get_embedding_dict('./glove.6B/glove.6B.%dd.txt'%embedding_dim)
    print ('Found %s word vectors.' % len(embedding_dict))
    max_features = len(word_index) + 1  # i.e. the number of words
    print ('Create embedding matrix.')
    embedding_matrix = get_embedding_matrix(word_index,embedding_dict,max_features,embedding_dim)

    print('Build model...')

    csv_logger = CSVLogger('training_report.csv',append=True)
    earlystopping = EarlyStopping(monitor='val_f1_measure', patience = 7, verbose=1, mode='max')
    checkpoint = ModelCheckpoint(filepath='best.hdf5',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 monitor='val_f1_score',
                                 mode='max')
    #sgd = SGD(lr=0.01, decay=0.0, momentum=0.9, nesterov=True)
    adam = Adam(lr=0.001,decay=1e-6,clipvalue=0.5)
    model = Sequential()
    model.add(Embedding(max_features,
                        embedding_dim,
                        weights=[embedding_matrix],
                        input_length=max_article_length,
                        trainable=False
                        #embeddings_constraint = maxnorm(2.),
                        #embeddings_regularizer = regularizers.l1(0.00001)
                       )
            )
    '''
    filters = 128
    kernel_size= 3

    model.add(Conv1D(filters,kernel_size,padding='same',strides=1))
    model.add(Dropout(0.2))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2,2,padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv1D(filters*2,kernel_size,padding='same',strides=1))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2,2,padding='same'))
    '''
    #model.add(GRU(128, dropout=0.3,activation='tanh',return_sequences=True))
    model.add(GRU(256, dropout=0.4,activation='tanh'))
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(label_num, activation='hard_sigmoid'))

    model.compile(loss='categorical_crossentropy',
                  optimizer = adam,
                  metrics=[f1_measure],
                  callbacks=[earlystopping,checkpoint]
                 )

    model.summary()
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epoch_num,
              validation_data =(x_test, y_test),
              callbacks=[csv_logger]
             )

    out = model.predict(x_test)
    '''
    model.fit(x_test, y_test,
            batch_size=batch_size,
            epochs=3,
            callbacks=[csv_logger]
     )
    '''
    acc = []
    accuracies = []
    threshold = np.arange(0.4,0.6,0.01)
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
    print(f1_score(y_test, y_pred, average='samples'))

    score = model.evaluate(x_train,y_train)

    print ('\nTrain Acc: ', score[1])
    model.save('my_model_4.h5')

if __name__=='__main__':
    main()

