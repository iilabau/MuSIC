from sklearn import datasets
from sklearn.model_selection import train_test_split
from iilab import MuSIC, normalize
from sklearn.metrics import accuracy_score

# prepare data
ds = datasets.load_digits()
X = ds.images
y = ds.target
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2) 

# train the model and test
mdl = MuSIC(scales=[1.50, 2.0, 2.5], epoch=5, verbose=1)
mdl.fit(X_train, y_train)
y_pred = mdl.predict(X_test, scalewise=True)

# print the results for each scale
for i in range(len(y_pred)-1):
    print('Scale %d: Acc=%f'%(i,accuracy_score(y_test,y_pred[i])))

# print the result of MuSIC model
print('MUSICv1: Acc=%f'%accuracy_score(y_test,y_pred[-1]))