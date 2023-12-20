# -*- coding: utf-8 -*-
"""polynomial_regression.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1eE4FrT33jThbYfwV3AoLnJOpdWiipFOq
"""

import pandas as pd

# Load the provided data files
dev_data = pd.read_csv('my_dev.csv')
train_data = pd.read_csv('my_train.csv')

# Display the first few rows of each dataset for an overview
dev_data.head(), train_data.head()

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Create 'OverallArea' feature
train_data['OverallArea'] = train_data['1stFlrSF'] + train_data['2ndFlrSF'] + train_data['TotalBsmtSF']
dev_data['OverallArea'] = dev_data['1stFlrSF'] + dev_data['2ndFlrSF'] + dev_data['TotalBsmtSF']

# Selecting features and target for the model
features_train = train_data[['OverallArea', 'LotArea']]
target_train = np.log(train_data['SalePrice'])
features_dev = dev_data[['OverallArea', 'LotArea']]
target_dev = np.log(dev_data['SalePrice'])

# Applying PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
features_train_poly = poly.fit_transform(features_train)
features_dev_poly = poly.transform(features_dev)

# Fit the model
model = LinearRegression()
model.fit(features_train_poly, target_train)

# Predict and evaluate
target_pred_train = model.predict(features_train_poly)
target_pred_dev = model.predict(features_dev_poly)
mse_train = mean_squared_error(target_train, target_pred_train)
mse_dev = mean_squared_error(target_dev, target_pred_dev)

mse_train, mse_dev, poly.get_feature_names_out(features_train.columns)
rmse_train = np.sqrt(mse_train)
rmse_dev = np.sqrt(mse_dev)
print("Root Mean Squared Error:", rmse_train,rmse_dev)







import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Assuming train_data and dev_data are already loaded

# Create 'OverallArea' feature
train_data['OverallArea'] = train_data['1stFlrSF'] + train_data['2ndFlrSF'] + train_data['TotalBsmtSF']
dev_data['OverallArea'] = dev_data['1stFlrSF'] + dev_data['2ndFlrSF'] + dev_data['TotalBsmtSF']

# Selecting features and target for the model
features_train = train_data[['OverallArea', 'LotArea']]
target_train = np.log(train_data['SalePrice'])  # Log transformation
features_dev = dev_data[['OverallArea', 'LotArea']]
target_dev = np.log(dev_data['SalePrice'])  # Log transformation

# Applying PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
features_train_poly = poly.fit_transform(features_train)
features_dev_poly = poly.transform(features_dev)

# Fit the model
model = LinearRegression()
model.fit(features_train_poly, target_train)

# Predict and evaluate
target_pred_train = model.predict(features_train_poly)
target_pred_dev = model.predict(features_dev_poly)

# Since targets are already log-transformed, RMSE is equivalent to RMSLE
rmsle_train = np.sqrt(mean_squared_error(target_train, target_pred_train))
rmsle_dev = np.sqrt(mean_squared_error(target_dev, target_pred_dev))

print("Root Mean Squared Logarithmic Error (Train):", rmsle_train)
print("Root Mean Squared Logarithmic Error (Dev):", rmsle_dev)