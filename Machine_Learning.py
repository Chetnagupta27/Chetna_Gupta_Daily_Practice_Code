#!/usr/bin/env python
# coding: utf-8

# ##Loading the data

# In[1]:


from sklearn import datasets


# In[5]:


boston=datasets.load_boston()
type(data)


# In[3]:


data


# In[6]:


x=boston.data
y=boston.target


# In[7]:


x.shape


# In[8]:


type(x)


# In[13]:


import pandas as pd
df=pd.DataFrame(x)
print(boston.feature_names)
df.columns=boston.feature_names
df.describe()


# In[15]:


boston.DESCR


# model selection-- It devides the data into parts 

# In[22]:


from sklearn import model_selection
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# Linear Regression

# In[23]:


from sklearn.linear_model import LinearRegression
alg1=LinearRegression()


# In[24]:


alg1.fit(x_train,y_train)


# In[26]:


y_pred=alg1.predict(x_test)

##comparing y_pred and y_test
##Here y_pred is what our x_test is predicting and y_test is actual output


# In[32]:


import matplotlib.pyplot as plt
plt.scatter(y_pred,y_test,color="pink")
plt.axis([0,40,0,40])
plt.show()


# In[6]:


import numpy as np
data=np.array("Drugs_package.csv")


# In[9]:





# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[5]:


df=pd.read_csv('House_prices.csv')


# In[6]:


df


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel("area(sqr ft)")
plt.ylabel("price(US$)")
plt.scatter(df.area,df.price,color="red",marker='*')


# In[13]:


reg=linear_model.LinearRegression()
reg.fit(df[["area"]],df.price)


# In[23]:


reg.predict(5000)


# In[16]:


reg.coef_


# In[18]:


reg.intercept_


# In[49]:



78.42123288*3200+388838.35616438347


# In[25]:


area=pd.read_csv('areas.csv')


# In[50]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel("area(sqr ft)")
plt.ylabel("price(US$)")
plt.scatter(df.area,df.price,color="red",marker='*')
plt.plot(df.area,reg.predict(df[["area"]]),color="blue")


# In[26]:


area


# In[32]:


p=reg.predict(area)


# In[33]:


area['prices']=p


# In[38]:


area.to_csv('areas.csv',index=False)


# In[39]:


area


# In[52]:


df=pd.read_csv("house_prices_02.csv")
df


# In[53]:


import math
median_bedrooms=math.floor(df.bedrooms.median())
median_bedrooms


# In[54]:


df.bedrooms=df.bedrooms.fillna(median_bedrooms)
df


# In[55]:


reg=linear_model.LinearRegression()
reg.fit(df[["area","bedrooms","age"]],df.price)


# In[56]:


reg.coef_


# In[57]:


reg.intercept_


# In[59]:


reg.predict([[3000,3,15]])


# In[ ]:




