import numpy as np
import pandas as pd
import os


# print(os.getcwd())


def split_data(data_frame):
  return data_frame.iloc[0:699] #Splitting into 7:2

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
  return pd.get_dummies(df[column_name], prefix=column_name)

# use this whenever there is a need of binary encodings to 0 or 1
def convert_to_binary(df, column_name):
  a , b = df[column_name].unique()
  dict_new = {a: 0, b: 1}
  return df[column_name].map(dict_new)
  
def norm_x(column, max, min):
  column_array = column.values
  for i in range(len(column_array)):
    column_array[i] = (column_array[i] - min)/(max - min)
  return pd.Series(column_array)

def main():
  df_original = pd.read_csv('train.csv')
  df = split_data(df_original)
  y = df['Survived']
  print(df.head())
  df['Pclass'] = convert_to_object(df, 'Pclass')
  remove_null(df['Age'], df['Age'].mean())
  remove_null(df['Embarked'], 'S')
  print(df.isnull().sum())

if __name__ == '__main__':
  main()
