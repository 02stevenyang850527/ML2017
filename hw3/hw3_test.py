import pandas as pd
import numpy as np
import sys
from keras.models import load_model

test  = pd.read_csv(sys.argv[1])
test_feature = test.feature.str.split(' ')
test_feature = test_feature.tolist()
test_feature = np.array(test_feature, dtype=float)
test_feature = test_feature/255
test_feature = test_feature.reshape(test_feature.shape[0],48,48,1)

model = load_model('my_model.h5')

prediction = model.predict_classes(test_feature,batch_size=100,verbose=1)

of = open(sys.argv[2],'w')
out = 'id,label\n'

for i in range(len(prediction)):
    out = out + str(i) + ',' + str(prediction[i]) + '\n'

of.write(out)
of.close
