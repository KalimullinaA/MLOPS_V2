import pandas as pd
import os

df = pd.read_csv('data/df.csv')

df.drop(columns=['tourists', 'green_waste', 'rate','other', 'food', 'plastic', 'glass', 'waste_recycling', 'tourists'], inplace=True)

os.makedirs(os.path.join("data", "stage1"),exist_ok=True)

df.to_csv('data/stage1/new_df.csv', index=False)