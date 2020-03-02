# pandas to import the dataset
# sklearn to perform the splitting

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


#Loading the Dataset
data=pd.read_csv('forestfires.csv')
data.head()


#splitting
y=data.temp
x=data.drop('temp',axis=1)