# pandas to import the dataset
# sklearn to perform the splitting

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris=load_iris()
x,y=iris.data,iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.5,test_size=0.5,random_state=123)

y_test
y_train

# Plotting of Train and Test Set
# We fit our model on the train data to make predictions on it. 
# Let’s import the linear_model from sklearn, apply linear regression to the dataset, # and plot the results.
from sklearn.linear_model import LinearRegression as lm
model=lm().fit(x_train,y_train)
predictions=model.predict(x_test)
import matplotlib.pyplot as plt
plt.scatter(y_test,predictions)
plt.xlabel('True values')
plt.ylabel('Predictions')
plt.show()