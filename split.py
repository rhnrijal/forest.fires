# pandas to import the dataset
# sklearn to perform the splitting

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


# loading the Dataset
data=pd.read_csv('forestfires.csv')
data.head()


# splitting
# split this data into labels and features.
# Using features, we predict labels. 
# using features (the data we use to predict labels), we predict labels (the data we want to predict)
y=data.temp
x=data.drop('temp',axis=1)


# training
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
x_train.head()

x_train.shape