import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

# loading data
df=pd.read_csv('.\Irislec.csv')
print(df.head())


# data too large lets look at the shape 
# used flush to make the data to be immediatelety
print("Dataset length ::", len(df), flush=True)
print("Dataset Shape :: ",df.shape, flush=True)

# to build and train our tree

# separate target values 
x=df.values[:,0:4] 
y=df.values[:,5]
# splitting the dataset into train and test  30percent used 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)
# function to perform  trainingwith entropy 
clf_entropy=DecisionTreeClassifier(criterion='entropy',random_state=100,max_depth=3,min_samples_leaf=5)
clf_entropy.fit(x_train,y_train)
print(clf_entropy.fit(x_train,y_train))
print(clf_entropy)


# making a preddiction
# x-test would be the  new flowers we want 
y_pred_en=clf_entropy.predict(x_test)
print(y_pred_en)

# how accuraete is the  model ?
print("Accuracy is", accuracy_score(y_test,y_pred_en)*100)