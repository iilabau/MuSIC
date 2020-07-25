import numpy as np
import os, cv2
from iilab import MuSIC, normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import np_utils
from sklearn.utils import shuffle

# read image data
n_channel = 1
data_path = 'dataset'
imgdata_list = []
for dir_i in os.listdir(data_path):
    img_list = os.listdir(data_path + '/' + dir_i)
    for imgfile in img_list:
        img = cv2.imread(data_path + '/' + dir_i + '/' + imgfile)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_resized = normalize(img, channel=n_channel, row=45, col=135)
        imgdata_list.append(img_resized)

X = np.array(imgdata_list)
X = X.astype('float32')
X /= 255


# assign the class lebel
y = np.ones((X.shape[0],), dtype='int')
y[0:50] = 0
y[50:100] = 1
y[100:150] = 2  


# split data into training and test
#X, y = shuffle(X, y, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2) 


# train the model and test
mdl = MuSIC(scales=[0.5,0.75,0.90,1.0,1.1], epoch=3, verbose=1)
mdl.fit(X_train, y_train)
y_pred = mdl.predict(X_test, scalewise=True)

# print the results for each scale
for i in range(len(y_pred)-1):
    print('Scale %d: Acc=%f'%(i,accuracy_score(y_test,y_pred[i])))

# print the result of MuSIC model
print('MUSICv1: Acc=%f'%accuracy_score(y_test,y_pred[-1]))