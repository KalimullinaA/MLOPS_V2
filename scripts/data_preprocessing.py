import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np
import yaml
from sklearn.pipeline import Pipeline # Pipeline.Не добавить, не убавить
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler # Импортируем нормализацию и One-Hot Encoding от sklearn
from sklearn.compose import ColumnTransformer # т.н. преобразователь колонок


df = pd.read_csv('data/stage1/new_df.csv')

params = yaml.safe_load(open("params.yaml"))["split"]
p_split_ratio = params["split_ratio"]

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=p_split_ratio, random_state=120)

os.makedirs(os.path.join("data", "stage2"),exist_ok=True)

with open('data/stage2/X_train.npy', 'wb') as f:
    np.save(f, X_train)
with open('data/stage2/X_test.npy', 'wb') as f:
    np.save(f, X_test)
with open('data/stage2/y_train.npy', 'wb') as f:
    np.save(f, y_train)
with open('data/stage2/y_test.npy', 'wb') as f:
    np.save(f, y_test)