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
# Temp is a label to predict temperatures in y
# we use the drop() function to take all other data in x. 
# Then, we split the data.
y=data.temp
x=data.drop('temp',axis=1)


# training
# We usually split the data around 20%-80% between testing and training stages. 
# Under supervised learning, we split a dataset into a training data and test data in Python ML.
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

x_train.head()
x_train.shape # output (413, 12)

x_test.head()
x_test.shape # output (104, 12)

# The line test_size=0.2 suggests that the test data should be 20% of the dataset and the rest should be train data. 
# With the outputs of the shape() functions, you can see that we have 104 rows in the test data and 413 in the training data.