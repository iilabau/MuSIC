
import numpy as np
import os, cv2, h5py
from keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers.convolutional import Convolution2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD, RMSprop, Adam
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from cycler import cycler
import keras
import tensorflow as tf
config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
sess = tf.compat.v1.Session(config=config) 
keras.backend.set_session(sess)


# MuSIC version 2 model
class musicv2:
    def __init__(self, scales=[(25,25),(50,50)], channel=1, k=0.01, epoch=5, val_size=0.15, verbose=0):
        self.scales = scales
        self.channel = channel
        self.k = k
        self.epoch = epoch
        self.val_size = val_size
        self.verbose = verbose
        self.mdl = []        
        self.noc = 0    
        

    def read_data(self, path, sf):
        X = []
        y = []
        w = []
        
        cls = 0
        for cls_lbl in os.listdir(path):            
            img_list = os.listdir(path + '/' + cls_lbl)
            for imgfile in img_list:
                img = cv2.imread(path + '/' + cls_lbl + '/' + imgfile)                
                
                if self.channel == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                org_area = img.shape[0]*img.shape[1]                                
                img, obj_area = self.normalize(img, row=sf[0], col=sf[1])                
                #sz = img.shape
                #weight = 1 - self.k * abs(sf[0]*sf[1] - sz[0]*sz[1])/(sz[0]*sz[1])                
                weight = 1 - self.k * abs(obj_area - org_area)/org_area
                
                X.append(img)                
                y.append(cls)
                w.append(weight)
            cls = cls + 1
        
        X = np.array(X)
        X = X.astype('float32')        
        X /= 255
        
        if len(X[0].shape) == 2:
            X = np.expand_dims(X, axis=3)
                
        y = np_utils.to_categorical(y, cls)
        self.noc = cls
        
        return X, y, w

    
    def normalize(self, img, row, col):    
        sz = img.shape
        r, c = sz[0],sz[1]
                    
        # pad to match aspect ratio    
        ar_org = c/r
        ar_new = col/row
        if ar_new > ar_org:        
            r_int = r
            c_int = round(c*ar_new/ar_org)
        else:
            r_int = round(r*ar_org/ar_new)
            c_int = c
        
        if self.channel == 1:
            img_new = np.zeros((r_int,c_int), dtype='uint8')    
            img_new[:r,:c] = img
        else:            
            img_new = np.zeros((r_int,c_int,3), dtype='uint8')    
            img_new[:r,:c,:] = img
            
        # resize to match target dimension
        img_new = cv2.resize(img_new, (col,row))    
       # cv2.imwrite('F:/script identification dataset/upscale/horizontal.png', img_new)
        
        # compute no. of object pixels in normalized image
        obj_area = int((row*col)*(r*c)/(r_int*c_int))
        
        return img_new, obj_area

    
    def fit(self, path):
        for sf in self.scales:
            X, y, _ = self.read_data(path, sf)            

            sz = X[0].shape                        
            print('Training begins for scale [%dx%d]'%(sz[0],sz[1]))
            print('----------------------------------')
            
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
            model.add(Convolution2D(16, (5, 5)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            
            model.add(Flatten())
            model.add(Dropout(0.4))
            model.add(Dense(128))
            model.add(Activation('relu'))
            model.add(Dense(50))
            model.add(Activation('relu'))
            model.add(Dense(self.noc))
            model.add(Activation('softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["acc"])
            
            # Viewing model_configuration
            print("\nCNN MODEL SUMMARY:\n ")
            model.summary()
            print("\nCNN MODEL CONFIGURATION:\n")
            model.get_config()
            model.layers[0].get_config()
            model.layers[0].input_shape
            model.layers[0].output_shape
            model.layers[1].get_weights()
            np.shape(model.layers[0].get_weights()[0])
            model.layers[1].trainable
            
            X_trn, X_vld, y_trn, y_vld = train_test_split(X, y, test_size=self.val_size)
            #X_trn, X_vld, y_trn, y_vld = train_test_split(X, y, test_size=self.val_size, random_state=5)
            
            history = model.fit(X_trn, y_trn,
              batch_size=10,
              epochs=self.epoch,
              verbose=self.verbose,
              validation_data=(X_vld, y_vld),
              )
            plt.style.use('Solarize_Light2')
            plt.plot(history.history['acc'], color='blue')
            plt.plot(history.history['val_acc'], color='magenta')
            plt.title('Training vs.validation accuracy')
            plt.ylabel('Accuracy', fontsize = 14, color='black')
            plt.xlabel('Epoch', fontsize=14, color='black')
            plt.legend(['train', 'validation'],loc='lower right')
            plt.show()

            plt.style.use('Solarize_Light2')
            plt.plot(history.history['loss'], color='blue')
            plt.plot(history.history['val_loss'], color='magenta')
            plt.title('Training vs.validation loss')
            plt.ylabel('Loss', fontsize=14, color = 'black')
            plt.xlabel('Epoch', fontsize = 14, color= 'black')
            plt.legend(['train', 'validation'],loc='upper right')
            plt.show()
            self.mdl.append(model)                        
            print('')
            
        return self    

    
    def predict(self, path):                
        scores = []
        for s in range(len(self.scales)):
            X, y, w = self.read_data(path, self.scales[s])
            y_true = np.argmax(y, axis=1)
            
            if scores == []:                
                scores = np.zeros((len(X), self.noc), dtype=float)                

            y_pred = self.mdl[s].predict_classes(X)                            
            for i in range(len(X)):
                scores[i,y_pred[i]] += w[i]        
            
            acc = accuracy_score(y_true, y_pred)
            Pr = precision_score(y_true, y_pred, average='macro')
            Re = recall_score(y_true, y_pred, average='macro')
            F1 = f1_score(y_true, y_pred, average='macro')
            
            print('Scale [%dx%d] Acc = %f'%(self.scales[s][0],self.scales[s][1],acc))
            print('Scale [%dx%d] Precision = %f'%(self.scales[s][0],self.scales[s][1],Pr)) 
            print('Scale [%dx%d] Recall = %f'%(self.scales[s][0],self.scales[s][1],Re)) 
            print('Scale [%dx%d] F-measure = %f'%(self.scales[s][0],self.scales[s][1],F1))
            print('.............')
        
        y_pred = np.argmax(scores, axis=1)                
        acc=accuracy_score(y_true, y_pred)
        Pr = precision_score(y_true, y_pred, average='macro')
        Re = recall_score(y_true, y_pred, average='macro')
        F1 = f1_score(y_true, y_pred, average='macro')
        
        print('MuSICv2 Acc = %f'%(acc))
        print('MuSICv2 Precision = %f'%(Pr))
        print('MuSICv2 Recall = %f'%(Re))
        print('MuSICv2 F-measure = %f'%(F1))
        mat = confusion_matrix(y_true, y_pred)
        axes = sns.heatmap(mat, square= True, annot = True,fmt = 'd',cbar=True, cmap = plt.cm.Pastel2)
        class_labels = [ 'Arabic','Bangla', 'Hindi', 'Latin', 'Symbol']
        
        tick_marks = np.arange(len(class_labels))+0.5
        
        axes.set_xticks(tick_marks)
        axes.set_xticklabels(class_labels, rotation=90)
        axes.set_yticks(tick_marks)
        axes.set_yticklabels(class_labels, rotation=0)
        
        axes.set_xlabel('Actual')
        axes.set_ylabel('Prediction')
        axes.set_title('Confusion matrix of ICDAR 2019-MLT')
        
        return y_pred, y_true   





# MuSIC old version
class MuSIC:
    def __init__(self, scales=[0.9,1.0,1.1], k=0.01, epoch=25, val_size=0.15, verbose=0):
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
