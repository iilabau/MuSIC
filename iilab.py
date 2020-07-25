import cv2, h5py
import numpy as np
from keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers.convolutional import Convolution2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD, RMSprop, adam


class MuSIC:
    def __init__(self, scales=[0.9,1.0,1.1], k=0.01, epoch=25, val_size=0.2, verbose=0):
        self.scales = scales
        self.k = k
        self.epoch = epoch
        self.val_size = val_size
        self.verbose = verbose
        self.mdl = []
        self.weights = []
        self.noc = 0
        

    def fit(self, X_train, y_train):        
        self.noc = len(np.unique(y_train))
        
        if len(X_train[0].shape) == 2:
            X_train = np.expand_dims(X_train, axis=3)
        y_train = np_utils.to_categorical(y_train, self.noc)
        
        for sf in self.scales:                        
            X_scaled, w = self.rescale(X_train, sf)
            self.weights.append(w)
            sz = X_scaled[0].shape
            print('\n\nTraining begins for Scale %.2f [%dx%d]'%(sf,sz[0],sz[1]))
            print('---------------------------------------')
            
            # create the CNN model
            model = Sequential()            
            model.add(Convolution2D(32, (5, 5), padding='same', input_shape=sz))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            #model.add(Dropout(0.2))
            model.add(Convolution2D(32, (5, 5)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            #model.add(Dropout(0.4))
            model.add(Flatten())
            model.add(Dense(128))
            model.add(Activation('relu'))
            model.add(Dense(50))
            model.add(Activation('relu'))
            model.add(Dense(self.noc))
            model.add(Activation('softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["acc"])
            
            #X_trn, X_vld, y_trn, y_vld = train_test_split(X_train, y_train, test_size=self.val_size)
            X_trn, X_vld, y_trn, y_vld = train_test_split(X_scaled, y_train, 
                test_size=self.val_size, random_state=5)
            
            mdl_out = model.fit(X_trn, y_trn,
              batch_size=250,
              epochs=self.epoch,
              verbose=self.verbose,
              validation_data=(X_vld, y_vld),
              )
        
            self.mdl.append(model)
            
        return self

    
    def rescale(self, X, sf):
        n_sample = len(X)
        X_scaled = []

        # determine the new scale                
        sz = np.array(X[0].shape)
        r, c, channel = sz[0], sz[1], sz[2]        
        r_new = int(r * sf)
        c_new = int(c * sf)
        
        # prepare the scaled samples
        for i in range(n_sample):            
            img = X[i,:,:,:]
            img = cv2.resize(img, (c_new, r_new))
            img = img.reshape(r_new, c_new, channel)
            X_scaled.append(img)        
        X_scaled = np.array(X_scaled)
        
        # calculate the scale weight
        w = 1 - self.k * abs(r_new*c_new - r*c)/(r*c)        
        
        return X_scaled, w

    
    def predict(self, X_test, scalewise=False):
        pred_lst = []
        
        if len(X_test[0].shape) == 2:
            X_test = np.expand_dims(X_test, axis=3)
        
        if self.verbose:
            print('Weights of scales: ' + str(self.weights))
            
        for i in range(len(self.scales)):        
            X_scaled, _ = self.rescale(X_test, self.scales[i])     
            
            y_pred = self.mdl[i].predict_classes(X_scaled)
            pred_lst.append(list(y_pred))
        
        for i in range(len(X_test)):                        
            counts = np.zeros(self.noc)
            for j in range(len(self.scales)):
                counts[pred_lst[j][i]] += self.weights[j]
            y_pred[i] = np.argmax(counts)                        
        
        if scalewise:
            pred_lst.append(list(y_pred))
            return pred_lst
        else:
            return y_pred    



def normalize(img, channel=1, row=50, col=50):    
    sz = img.shape
    r, c = sz[0],sz[1]
        
    if channel == 1 and len(sz) != 2:        
        input('ERROR: Gray image expected as number of channel=%d'%channel)
    if channel != 1 and len(sz)==2:        
        input('ERROR: Color image expected as number of channel=%d'%channel)

    # pad to match aspect ratio    
    ar_org = c/r
    ar_new = col/row
    if ar_new > ar_org:        
        r_int = r
        c_int = round(c*ar_new/ar_org)
    else:
        r_int = round(r*ar_org/ar_new)
        c_int = c
    
    if channel == 1:
        img_new = np.zeros((r_int,c_int), dtype='uint8')    
        img_new[:r,:c] = img
    else:            
        img_new = np.zeros((r_int,c_int,3), dtype='uint8')    
        img_new[:r,:c,:] = img
        
    # resize to match target dimension
    img_new = cv2.resize(img_new, (col,row))    
    #cv2.imwrite('normalized.png', img_new)
    
    return img_new