import numpy as np
import pandas as pd
# from sklearn import preprocessing
# import math
import os
import sys
print(sys.path)
import data_manipulation as dm
import Logistic_Regression as log_r


# read_csv
df = pd.read_csv('train.csv')


# convert column of data_frame into object type
df['Pclass'] = dm.convert_to_object(df, 'Pclass')

# remove null values from the dataframe
dm.remove_null(df['Age'], df['Age'].mean())
dm.remove_null(df['Embarked'], 'S')
# print(df.isnull().sum())

# one hot encoding to a certain data
df = pd.concat([df,dm.one_hot_encoding(df, 'Pclass')], axis =1)
df.pop('Pclass')
df = pd.concat([df,dm.one_hot_encoding(df, 'Embarked')], axis =1)
df.pop('Embarked')


# create binary
df['Sex'+'_binary'] = dm.convert_to_binary(df, 'Sex')
df.pop('Sex')

# Calculate the features
features =df[['Age','SibSp','Parch','Fare','Sex_binary','Pclass_1','Pclass_2','Pclass_3','Embarked_C','Embarked_Q','Embarked_S']]
# print(features.describe())


# Normalize the data set
random_var0= dm.norm_x(features['Age'], features['Age'].max(), features['Age'].min())
features.replace(features['Age'],random_var0)

random_var1= dm.norm_x(features['SibSp'], features['SibSp'].max(), features['SibSp'].min())
features.replace(features['SibSp'],random_var1)

random_var2= dm.norm_x(features['Parch'], features['Parch'].max(), features['Parch'].min())
features.replace(features['Parch'],random_var2)

random_var3= dm.norm_x(features['Fare'], features['Fare'].max(), features['Fare'].min())
features.replace(features['Fare'],random_var3)

# split_data into training and testing
df_train = dm.split_data(features)
y = df['Survived'].iloc[0:699] #label


x = df_train.values
m = x.shape[0]
bias = np.ones([m,1])
x = np.hstack((bias, x))
theta = np.zeros(x.shape[1])
y_hat = log_r.activation(np.dot(x,theta))
y = y.values
alpha = 0.005

theta_out = log_r.gradient_descent(x, y, theta, alpha, m, y_hat)
for i in range(10000):
  h1 = log_r.activation(np.dot(x,theta_out))
  theta_out = log_r.gradient_descent(x, y, theta_out, alpha, m, h1)
  loss_out = log_r.loss(x, y, theta_out, m,h1)
  if i%5000==0:
    print(loss_out)
  if i ==9999:
    theta_final = theta_out


# Testing the remaining dataset and Prediction

# getting the remaining data
df_test = features.iloc[700:]

x_test = df_test.values
m_test = x_test.shape[0]
x_test = np.hstack((np.ones([m_test,1]), x_test))

y_predict = log_r.activation(np.dot(x_test,theta_final))
for i in range(len(y_predict)):
  if y_predict[i] >= 0.5:
    y_predict[i] = 1
  else:
    y_predict[i] = 0

print(y_predict)
print(len(y_predict))

#test accuracy .////////////////////////////////////
y_actual = df['Survived'].iloc[700:].values
count = 0
for i in range(len(y_actual)):
    if (predict_test[i] == y_test.values[i]):
        count = count +1

print(count)
accuracy = (count* 100) / (len(y_predict)) 
print(accuracy)

print("-------------------------------------------------------------------------")
print(theta_final)








