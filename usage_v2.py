from iilab import musicv2

# dimension of scales
dims = [(45,90), (50,100)]

mdl = musicv2(scales=dims, epoch=5, channel=1, verbose=1)

# path for training data
mdl.fit(path='F:/ICDAR2019_WORDLEVEL/training')

# path for test data       
y_pred, y_true = mdl.predict(path='F:/ICDAR2019_WORDLEVEL/validation_set')  
