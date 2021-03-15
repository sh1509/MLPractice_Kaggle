import numpy as np
import pandas as pd
import os


# print(os.getcwd())
df_original = pd.read_csv('train.csv')

def split_data(data_frame):
  return data_frame.iloc[0:699] #Splitting into 7:2


df = split_data(df_original)
y = df['Survived']
print(df.head())

def generate_features(df):
  pass

# convert the data types to object
def convert_to_object(df, column_name):
  return df[column_name].astype('object')

# remove null values from the data for each column
def remove_null(df, value):
  return df.fillna(value, inplace=True)

# generating one hot encoding for object of more than two values
def one_hot_encoding(df, column_name):
  return pd.concat([df,pd.get_dummies(df[column_name], prefix=column_name)], axis =1)

# use this whenever there is a need of binary encodings to 0 or 1
def convert_to_binary(df, column_name):
  a , b = df[column_name].unique()
  dict_new = {'a': 0, 'b': 1}
  df[column_name+'_binary'] = df[column_name].map(dict_new)
  df.pop(column_name)
  return df



df['Pclass'] = convert_to_object(df, 'Pclass')
remove_null(df['Age'], df['Age'].mean())
remove_null(df['Embarked'], 'S')
print(df.isnull().sum())
