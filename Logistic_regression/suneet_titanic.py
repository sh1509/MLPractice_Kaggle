import numpy as np
import pandas as pd
# from sklearn import preprocessing
# import math
import os

# import that python file
#from tanya_pynb import x_feature, y_target



# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/
# %cd /content/drive/MyDrive/Machine\ Learning\ Practice/
# %cd Linear Regression/

df_original = pd.read_csv('train.csv')

def split_data(data_frame):
  return data_frame.iloc[0:699] #Splitting into 7:2

df = split_data(df_original)
y = df['Survived']

def generate_features(df):
  pass

def one_hot_encoding(df):
  pass

df.loc[:,['Pclass']]=df.loc[:,['Pclass']].astype('object')

df['Embarked'].fillna("S", inplace=True)

df['Age'].fillna(df['Age'].mean(), inplace=True)
df.isnull().sum()

Sexdict = {'male': 0, 'female': 1}
df['Sex_binary']=df['Sex'].map(Sexdict)
df.pop('Sex')
df.head()

df = pd.concat([df,pd.get_dummies(df['Pclass'], prefix='Pclass')], axis =1)
df = pd.concat([df,pd.get_dummies(df['Embarked'], prefix='Embarked')], axis =1)
df.pop('Pclass')
df.pop('Embarked')
df.head()

features =df[['Age','SibSp','Parch','Fare','Sex_binary','Pclass_1','Pclass_2','Pclass_3','Embarked_C','Embarked_Q','Embarked_S']]
features.describe()

# scaler = preprocessing.StandardScaler().fit(features.values)
# features_scaled = scaler.transform(features.values)
# new_features=pd.DataFrame(features_scaled)
# new_features.describe()

def norm_x(column, max, min):
  column_array = column.values
  for i in range(len(column_array)):
    column_array[i] = (column_array[i] - min)/(max - min)
  return pd.Series(column_array)

random_var0= norm_x(features['Age'], features['Age'].max(), features['Age'].min())
features.replace(features['Age'],random_var0)

random_var1= norm_x(features['SibSp'], features['SibSp'].max(), features['SibSp'].min())
features.replace(features['SibSp'],random_var1)

random_var2= norm_x(features['Parch'], features['Parch'].max(), features['Parch'].min())
features.replace(features['Parch'],random_var2)

random_var3= norm_x(features['Fare'], features['Fare'].max(), features['Fare'].min())
features.replace(features['Fare'],random_var3)

def activation(z):
  return 1/(1 + np.exp(-z))

def gradient_descent(x, y, theta,alpha, m, h1):
    # h_ = activation(np.dot(x,theta))
    grad = np.dot((h1 - y), x) / m
    theta = theta - alpha*grad
    return theta

# Hey Yo! Can you do it?
# sjsj 🐔🐣🐧🐛
# I think yes. yeah i can now. Cool! First duvidha solv l I'll lloligkdkdbrxutxtoxdg😎🥺😅😂🤦🏻‍♀️😀Ll
# Dm pun con bhi dekh sakte hai. Kafi mast hai. Infact we can use every feature of vs code including notebook jo
# tujhe pasand hai. So lets go, lets go
def loss(x, y, theta, m,h):
  loss1 = np.dot(np.log(h),y)
  loss2 = np.dot((np.log(1-h)), (1-y))
  loss_out = -1*(loss1+ loss2)/m
  return loss_out

x = features.values
m = x.shape[0]
bias = np.ones([m,1])
x = np.hstack((bias, x))
theta = np.zeros(x.shape[1])

y_hat = activation(np.dot(x,theta))
y = y.values
alpha = 0.005

theta_out = gradient_descent(x, y, theta, alpha, m, y_hat)
for i in range(10000):
  h1 = activation(np.dot(x,theta_out))
  theta_out = gradient_descent(x, y, theta_out, alpha, m, h1)
  loss_out = loss(x, y, theta_out, m,h1)
  if i%5000==0:
    print(loss_out)
  if i ==9999:
    theta_final = theta_out

# Prediction

#df_test = pd.read_csv('test.csv')
df_test = df_original.iloc[700:]
df_test['Age'].fillna(df_test['Age'].mean(), inplace=True)

Sexdict = {'male': 0, 'female': 1}
df_test['Sex_binary']=df_test['Sex'].map(Sexdict)
df_test = pd.concat([df_test,pd.get_dummies(df_test['Pclass'], prefix='Pclass')], axis =1)
df_test = pd.concat([df_test,pd.get_dummies(df_test['Embarked'], prefix='Embarked')], axis =1)
df_test.pop('Sex')
df_test.pop('Pclass')
df_test.pop('Embarked')

features_test =df_test[['Age','SibSp','Parch','Fare','Sex_binary','Pclass_1','Pclass_2','Pclass_3','Embarked_C','Embarked_Q','Embarked_S']]

random_var00= norm_x(features_test['Age'], features_test['Age'].max(), features_test['Age'].min())
features_test.replace(features_test['Age'],random_var00)

random_var01= norm_x(features_test['SibSp'], features_test['SibSp'].max(), features_test['SibSp'].min())
features_test.replace(features_test['SibSp'],random_var01)

random_var02= norm_x(features_test['Parch'], features_test['Parch'].max(), features_test['Parch'].min())
features_test.replace(features_test['Parch'],random_var02)

random_var03= norm_x(features_test['Fare'], features_test['Fare'].max(), features_test['Fare'].min())
features_test.replace(features_test['Fare'],random_var03)

x_test = features_test.values
m_test = x_test.shape[0]
x_test = np.hstack((np.ones([m_test,1]), x_test))

y_predict = activation(np.dot(x_test,theta_final))
for i in range(len(y_predict)):
  if y_predict[i] >= 0.5:
    y_predict[i] = 1
  else:
    y_predict[i] = 0

print(y_predict)
print(len(y_predict))

#test accuracy
y_actual = df_test['Survived'].values
correct_y = np.count_nonzero(np.all([y_predict, y_actual], axis = 0))
print(correct_y)
accuracy = (correct_y* 100) / (len(y_predict)) 
print(accuracy)

print("-------------------------------------------------------------------------")
print(theta_final)






