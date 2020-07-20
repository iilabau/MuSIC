import numpy as np
from iilab import MuSIC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# read image data
X = 1000x32x32
y = 1000x1

# split data into training and test
X, y = shuffle(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25) 

# train the model and test    
mdl = MuSIC(nos=1, sf=0.05, epoch=5)
mdl.fit(X_train, y_train)
y_pred = mdl.predict(X_test)   

# print the results
print(accuracy_score(y_test,y_pred))