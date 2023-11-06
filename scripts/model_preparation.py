import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle


with open('data/stage2/X_train.npy', 'rb') as f:
    X_train = np.load(f, allow_pickle=True)
with open('data/stage2/y_train.npy', 'rb') as f:
    y_train = np.load(f, allow_pickle=True)
with open('data/stage2/X_test.npy', 'rb') as f:
    X_test = np.load(f, allow_pickle=True)
with open('data/stage2/y_test.npy', 'rb') as f:
    y_test = np.load(f, allow_pickle=True)


penalty= 'l1' #@param [ "l2" , "l1", "none"]{type:"string"}
regularization = 7.5 #@param {type:"slider", min:0, max:10, step:0.05}

model = LogisticRegression(fit_intercept=True,
                            penalty=penalty,solver='liblinear',
                            C=regularization,
                            max_iter=10000)
model.fit(X_train, y_train)


with open("models/model.pkl", "wb") as m:
    pickle.dump(model, m)