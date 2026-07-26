# import libraries

from sklearn.datasets import load_breast_cancer
import pandas as pd

# load the dataset

data=load_breast_cancer()

# explore the dataset

print(data)
print(data.feature_names)
print(data.target_names)
print(data.target[:10])

# create DataFrame
df=pd.DataFrame(data.data,columns=data.feature_names)

# understand data

print(df.head())
print(df.shape)
print(df.info())
print(df.describe())

# create features X and labels y
X=df
y=data.target

# split the data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# check the split
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# import the model
from sklearn.linear_model import LogisticRegression

# create the model
model=LogisticRegression(max_iter=10000)

# train the model
model.fit(X_train,y_train)


y_pred=model.predict(X_test)
print(y_pred[:10])

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)
