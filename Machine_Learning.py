#!/usr/bin/env python
# coding: utf-8

# ##Loading the data

from sklearn import datasets
boston=datasets.load_boston()
type(data)

data

x=boston.data
y=boston.target

x.shape

type(x)


import pandas as pd
df=pd.DataFrame(x)
print(boston.feature_names)
df.columns=boston.feature_names
df.describe()

boston.DESCR


# model selection-- It devides the data into parts 


from sklearn import model_selection
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# Linear Regression

from sklearn.linear_model import LinearRegression
alg1=LinearRegression()

alg1.fit(x_train,y_train)

y_pred=alg1.predict(x_test)

##comparing y_pred and y_test
##Here y_pred is what our x_test is predicting and y_test is actual output

import matplotlib.pyplot as plt
plt.scatter(y_pred,y_test,color="pink")
plt.axis([0,40,0,40])
plt.show()


import numpy as np
data=np.array("Drugs_package.csv")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df=pd.read_csv('House_prices.csv')


df


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel("area(sqr ft)")
plt.ylabel("price(US$)")
plt.scatter(df.area,df.price,color="red",marker='*')
reg=linear_model.LinearRegression()
reg.fit(df[["area"]],df.price)
reg.predict(5000)


reg.coef_

reg.intercept_
78.42123288*3200+388838.35616438347

area=pd.read_csv('areas.csv')


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel("area(sqr ft)")
plt.ylabel("price(US$)")



p=reg.predict(area)



area['prices']=p




area.to_csv('areas.csv',index=False)

area




df=pd.read_csv("house_prices_02.csv")
df

import math
median_bedrooms=math.floor(df.bedrooms.median())
median_bedrooms


df.bedrooms=df.bedrooms.fillna(median_bedrooms)
df



reg=linear_model.LinearRegression()
reg.fit(df[["area","bedrooms","age"]],df.price)


reg.coef_




reg.intercept_


reg.predict([[3000,3,15]])






