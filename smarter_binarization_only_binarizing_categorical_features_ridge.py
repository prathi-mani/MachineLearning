# -*- coding: utf-8 -*-
"""Smarter binarization: Only binarizing categorical features -ridge

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HKw6r_C-M_jIXjya89GkFe-TXx7Oc5uD
"""

from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import RobustScaler
import numpy as np

dev_data = pd.read_csv('my_dev.csv')
train_data = pd.read_csv('my_train.csv')

Numerical_Headers =  [ 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',  'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']
Categorical_Headers = ['MSZoning','Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive',  'SaleType', 'SaleCondition']
# Separating features and labels

train_data['SalePrice'] = np.log(train_data['SalePrice'])
dev_data['SalePrice'] = np.log(dev_data['SalePrice'])


dev_data_label = dev_data['SalePrice']
train_data_label = train_data['SalePrice']

dev_data_features = dev_data.drop(['Id', 'SalePrice','Alley','GarageYrBlt','PoolQC','Fence','MiscFeature'], axis=1)
train_data_features = train_data.drop(['Id', 'SalePrice','Alley','GarageYrBlt','PoolQC','Fence','MiscFeature'], axis=1)


for column in Categorical_Headers:
    train_data_features[column] = train_data_features[column].astype(str)
    dev_data_features[column] = dev_data_features[column].astype(str)


#binarizing



# Define preprocessing for numerical and categorical data
num_processor = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', MinMaxScaler(feature_range=(0, 1)))
])

cat_processor = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(sparse_output=True, handle_unknown='ignore'))
])

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_processor, Numerical_Headers),
        ('cat', cat_processor, Categorical_Headers)
    ]
)

#preprocessor.fit(train_data_features)
##binary_train_data_features=preprocessor.transform(train_data_features)
#binary_dev_data_features= preprocessor.transform(dev_data_features)

# Define the Ridge regression model with the parameters you provided
alphas = np.linspace(1, 50, 50)
ridge_model = RidgeCV(alphas=alphas, cv=None, store_cv_values=True)

# Create a new pipeline with the preprocessor and the Ridge regression model
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('scaler', RobustScaler(with_centering=False)),
    ('ridge', ridge_model)
])

# Fit the full pipeline to your training data
X_train = train_data_features  # Assuming train_data_features is your training data
y_train = train_data_label  # Assuming train_data_label is your training labels
model = full_pipeline.fit(X_train, y_train)

# Extract and print the optimal alpha and the score
optimal_alpha = model.named_steps['ridge'].alpha_
best_score = np.sqrt(-model.named_steps['ridge'].best_score_)

print('Alpha:', optimal_alpha)
print('Score:', best_score)

# Fit the model with the binary (encoded) features
model = LinearRegression()

model.fit(binary_train_data_features, train_data_label)


train_pred_log = model.predict(binary_train_data_features)
dev_pred_log = model.predict(binary_dev_data_features)


# Calculate RMSLE
rmsle_train = np.sqrt(np.mean((train_pred_log - train_data_label) ** 2))
rmsle = np.sqrt(np.mean((dev_pred_log - dev_data_label) ** 2))
print("RMSLE on Dev Set: " + str(rmsle))
print("RMSLE on test Set: " + str(rmsle_train))