import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ds=pd.read_csv('Data.csv')
x=ds.iloc[:,:-1].values
y=ds.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)

plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,reg.predict(x_train),color='blue')
plt.title('Linear Regression Model - Training Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='blue')
plt.title('Linear Regression Model - Testing Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()