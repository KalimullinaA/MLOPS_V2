import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np

df = pd.read_csv('data/stage1/new_df.csv')

cat_columns = []
num_columns = []

for column_name in df.columns:
    if (df[column_name].dtypes == object):
        cat_columns +=[column_name]
    else:
        num_columns +=[column_name]


#Применим OneHotEncoder к категориальным признакам
ohe_cat_col = pd.get_dummies(df[cat_columns])
df = df.join(ohe_cat_col)
df.drop(columns=['id', 'code', 'Country', 'period'], inplace=True)


X, y = df.drop(columns = ['polution_clf']).values, df['polution_clf'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=120)

os.makedirs(os.path.join("data", "stage2"),exist_ok=True)

with open('data/stage2/X_train.npy', 'wb') as f:
    np.save(f, X_train)
with open('data/stage2/X_test.npy', 'wb') as f:
    np.save(f, X_test)
with open('data/stage2/y_train.npy', 'wb') as f:
    np.save(f, y_train)
with open('data/stage2/y_test.npy', 'wb') as f:
    np.save(f, y_test)
# X_train.to_csv('data/stage2/X_train.csv', index=False)
# X_test.to_csv('data/stage2/X_test.csv', index=False)
# y_train.to_csv('data/stage2/y_train.csv', index=False)
# y_test.to_csv('data/stage2/y_test.csv', index=False)