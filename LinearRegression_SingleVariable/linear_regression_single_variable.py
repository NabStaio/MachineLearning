import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

#Read and save dataset
df = pd.read_csv("house.csv")

#Plot the dataset
%matplotlib inline
plt.xlabel("area(sqr ft)")
plt.ylabel("price(US$)")
plt.scatter(df.area, df.price, color="red", marker="+")

#Create object linear regression
reg = linear_model.LinearRegression()
#Apply fitting of feature "area"
reg.fit(df[['area']],df.price)
#Predict the price of a new given area by y = m * x + q
reg.predict([[5000]])

#Coefficient of the linear equation m
#reg.coef_
#Intercept q
#reg.intercept_

#plot of the linear model prediction
%matplotlib inline
plt.xlabel("area(sqr ft)", fontsize=20)
plt.ylabel("price(US$)", fontsize=20)
plt.scatter(df.area, df.price, color="red", marker="+")
plt.plot(df.area, reg.predict(df[['area']]), color = 'blue')

#predict some given areas
#d = pd.read_csv("price_to_predict.csv")
#p = reg.predict(d)

#Create a new column in the dataset
#d['prices'] = p
#Create new csv file with new column
#d.to_csv("prediction.csv", index = False)