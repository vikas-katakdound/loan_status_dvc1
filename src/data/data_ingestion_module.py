import pandas as  pd
import numpy as np

df = pd.read_csv("C:\\Users\\Vikas\\COOKIE\\demo123\\data\\external\\loan_approval_dataset.csv")

df.columns = df.columns.str.strip()

x = df.drop(['loan_id', 'loan_status'], axis=1)
y = df['loan_status']

import yaml

test_size = yaml.safe_load(open('C:\\Users\\Vikas\\COOKIE\\demo123\\params.yaml','r'))['data_ingestion']['test_size']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, stratify=y, test_size=test_size)
train_data = pd.concat([x_train, y_train], axis=1)
test_data = pd.concat([x_test, y_test], axis = 1)
 
# save the data locally using os library
import os
 
data_path = os.path.join('C:\\Users\\Vikas\\COOKIE\\demo123\\data', 'C:\\Users\\Vikas\\COOKIE\\demo123\\raw')
 
os.makedirs(data_path) 
 
train_data.to_csv(os.path.join(data_path, 'train.csv'), index=False)
test_data.to_csv(os.path.join(data_path, 'test.csv'), index=False)