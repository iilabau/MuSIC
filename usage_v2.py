from iilab import musicv2

dims = [(45,90), (50,100)]
mdl = musicv2(scales=dims, epoch=5, channel=1, verbose=1)
mdl.fit(path='F:/ICDAR2019_WORDLEVEL/training')
       
y_pred, y_true = mdl.predict(path='F:/ICDAR2019_WORDLEVEL/validation_set')  