import numpy as np
import os, cv2
from iilab import MuSIC, normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import np_utils
from sklearn.utils import shuffle

# read image data
n_channel = 3
data_path = 'dataset'
imgdata_list = []
for dir_i in os.listdir(data_path):
    img_list = os.listdir(data_path + '/' + dir_i)
    for imgfile in img_list:
        input_img = cv2.imread(data_path + '/' + dir_i + '/' + imgfile)      
        input_img_resized = normalize(input_img, channel=n_channel, row=45, col=135)
        imgdata_list.append(input_img_resized)

X = np.array(imgdata_list)
X = X.astype('float32')
X /= 255
if n_channel == 1:
    X = np.expand_dims(X, axis=3)


# assign the class lebel
labels = np.ones((X.shape[0],), dtype='int')
labels[1:50] = 0
labels[51:100] = 1
labels[101:151] = 2
num_class = 3


# convert class lebel into 1 D using one-hot-encoding
y = np_utils.to_categorical(labels, num_class)    


# split data into training and test
#X, y = shuffle(X, y, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2) 


# train the model and test
scales=[0.90, 1.0, 0.95]
mdl = MuSIC(scales=scales, epoch=1, verbose=1)

mdl.fit(X_train, y_train)

scalewise_result = True
y_pred = mdl.predict(X_test, scalewise=scalewise_result)


# print the results
if scalewise_result:    
    for i in range(len(scales)):
        y_scale = y_pred[i]
        y_scale = np_utils.to_categorical(y_scale, num_class)
        print('Scale %d: Acc=%f'%(i,accuracy_score(y_test,y_scale)))

    y_final = y_pred[num_class]
    y_final = np_utils.to_categorical(y_final, num_class)
    print('MUSICv1: Acc=%f'%accuracy_score(y_test,y_final))
else:
    y_pred = np_utils.to_categorical(y_pred, num_class)
    print('MUSICv1: Acc=%f'%accuracy_score(y_test,y_pred))