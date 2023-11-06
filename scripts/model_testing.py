from sklearn.metrics import f1_score
import pandas as pd
import pickle
import json
import numpy as np


with open('data/stage2/X_test.npy', 'rb') as f:
    X_test = np.load(f, allow_pickle=True)
with open('data/stage2/y_test.npy', 'rb') as f:
    y_test = np.load(f, allow_pickle=True)
model = pickle.load(open('models/model.pkl', 'rb'))


y_pred = model.predict(X_test)
score = f1_score(y_test, y_pred, average="micro")


with open('evaluate/score.json', 'w') as f:
    json.dump({"score": score}, f)