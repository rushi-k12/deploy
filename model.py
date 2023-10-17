# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 19:49:09 2023

@author: rushi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('housing.csv')
data.head()
data.info()
data['mainroad'] = data['mainroad'].replace({'yes':1,'no':0}).astype(int)
data['guestroom'] = data['guestroom'].replace({'yes':1,'no':0}).astype(int)
data['basement'] = data['basement'].replace({'yes':1,'no':0}).astype(int)
data['hotwaterheating'] = data['hotwaterheating'].replace({'yes':1,'no':0}).astype(int)
data['airconditioning'] = data['airconditioning'].replace({'yes':1,'no':0}).astype(int)
data['prefarea'] = data['prefarea'].replace({'yes':1,'no':0}).astype(int)
data['furnishingstatus'] = data['furnishingstatus'].replace({'furnished':2,'semi-furnished':1, 'unfurnished':0 }).astype(int)
corr = data.corr()
corr
price_corr = data.corr()
plt.figure(figsize= (12,7))
sns.heatmap(price_corr,annot = True,  cmap='Spectral')
plt.title('Correlation HeatMap')
plt.show()
sns.displot(data['price'])
sns.boxplot(data['area'])
Q1 = data['price'].quantile(0.25)
Q3 = data['price'].quantile(0.75)
IQR = Q3 - Q1

IQR, Q1, Q3
upper_bound = Q3 + 3 * IQR
lower_bound = Q1 - 3 * IQR

upper_bound, lower_bound
data = data[(data['price'] <= upper_bound) & (data['price'] >=  lower_bound)]
Q1_area = data['area'].quantile(0.25)
Q3_area = data['area'].quantile(0.75)
IQR_area = Q3_area - Q1_area

IQR_area, Q1_area, Q3_area
upper_bound_area = Q3_area + 3 * IQR_area
lower_bound_area = Q1_area - 3 * IQR_area

upper_bound_area, lower_bound_area
data = data[(data['area'] <= upper_bound_area) & (data['area'] >=  lower_bound_area)]
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
x = data.drop(['price'], axis = 1)
y = data['price']
scalar = StandardScaler()

x_scale = scalar.fit_transform(x)
random = RandomForestRegressor()
reg = LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(x_scale,y, test_size= 0.2 , random_state=2)
reg.fit(x_train,y_train)
random.fit(x_train,y_train)
y_pred_random = random.predict(x_test)
y_pred_reg = reg.predict(x_test)
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_pred_reg))
print('MSE:', metrics.mean_squared_error(y_test, y_pred_reg))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_reg)))
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred_reg))
print(r2_score(y_test, y_pred_random))

import pickle
pickle.dump(reg , open('model.pkl' , 'wb'))
model = pickle.load(open('model.pkl' , 'rb'))
print(model.predict)