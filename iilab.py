import numpy as np


class MuSIC:
    def __init__(self, nos=3, sf=0.01, epoch=25):                        
        self.nos = nos
        self.sf = sf
        self.epoch = epoch
        
        return self
        
    def fit(self, X_train, y_train):  
        #code for training
        print(self.epoch)
        
        return self
    
    def predict(self, X_test):
        y_pred = np.zeros(X_test.shape[0], dtype=int)        
        
        return y_pred