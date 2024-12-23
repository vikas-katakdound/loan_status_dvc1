import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

# Step 1: Load the data
train_data = pd.read_csv('C:\\Users\\Vikas\\COOKIE\\demo123\\raw\\train.csv')
test_data = pd.read_csv('C:\\Users\\Vikas\\COOKIE\\demo123\\raw\\test.csv')

# Step 2: Separate features (X) and target variable (y)
X_train = train_data.drop(columns=['loan_status'])
y_train = train_data['loan_status']
X_test = test_data.drop(columns=['loan_status'])
y_test = test_data['loan_status']

# Step 3: One-Hot Encoding of categorical features (using pd.get_dummies)
X_train_encoded = pd.get_dummies(X_train, drop_first=False)  # drop_first=False keeps all categories
X_test_encoded = pd.get_dummies(X_test, drop_first=False)

# Step 4: Align Test Data with Train Data Columns (to ensure both datasets have the same columns)
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

X_train_encoded.fillna(0, inplace=True)
X_test_encoded.fillna(0, inplace=True)


# Step 5: MinMax Scaling for numerical columns (assuming all columns are numerical after encoding)
scaler = MinMaxScaler()

# Apply scaling to all columns in X_train_encoded and X_test_encoded
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

# Convert the scaled arrays back to DataFrames with the original column names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_encoded.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test_encoded.columns)

# Step 6: Recombine preprocessed features with target variable
train_preprocessed = pd.concat([X_train_scaled, y_train], axis=1)
test_preprocessed = pd.concat([X_test_scaled, y_test], axis=1)


# create a data/processed folder and save the data

data_path = os.path.join('C:\\Users\\Vikas\\COOKIE\\demo123\\data', 'C:\\Users\\Vikas\\COOKIE\\demo123\\processed')

os.makedirs(data_path)


# save to these created path

train_preprocessed.to_csv(os.path.join(data_path, 'train_processed.csv'), index=False)
test_preprocessed.to_csv(os.path.join(data_path, 'test_processed.csv'), index= False)