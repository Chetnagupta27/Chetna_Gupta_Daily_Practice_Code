#!/usr/bin/env python
# coding: utf-8

# ##Loading the data

from sklearn import datasets
boston=datasets.load_boston()
type(data)

data

x=boston.data#!/usr/bin/env python
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


import numpy as np
def gradient_descent(x,y):
    m_curr=b_curr=0
    iterations=1000
    for i in range(iterations):
        y_predicted=m_curr*x+b_curr
        md=-(2/n)*sum(x*(y-y_predicted))
        

x=np.array([1,2,3,4,5])
y=np.array([5,7,9,11,13])

gradient_descent(x,y)


# In[8]:


import pandas as pd
df=pd.read_csv("homeprices.csv")
df


# In[14]:


dummies=pd.get_dummies(df.town)
dummies


# In[1]:


merged=pd.concat(df,dummies)


# In[5]:


import numpy as np

def gradient_descent(x,y):
    m_curr=b_curr=0
    iterations=1000
    n=len(x)
    learning_rate=0.08
    
    for i in range(iterations):
        y_predicted=m_curr*x+b_curr
        cost=(1/n)*sum(val**2 for val in (y-y_predicted))
        md=-(2/n)*sum(x*(y-y_predicted))
        bd=-(2/n)*sum(y-y_predicted)
        m_curr=m_curr-learning_rate*md
        b_curr=b_curr-learning_rate*bd
        print("m {},b {},cost{},iteration{}".format(m_curr,b_curr,cost,i))
        
x=np.array([1,2,3,4,5])
y=np.array([5,7,9,11,13])

gradient_descent(x,y)


# In[13]:


import pickle
with open('model_pickle',"wb")as f:
    pickle.dump(model,f)
with open('model pickle',"rb")as f:
    mp=pickle.load(f)


# In[11]:


model=linear_model.LinearRegression()
model.fit(df[['area']],df.price)
model.coef_
model.intercept_
model.predict(5000)


# In[9]:


import pandas as pd
import numpy as np
from sklearn import linear_model


# In[10]:


df=pd.read_csv("homeprices.csv")
df.head()


# In[15]:


import pandas as pd
df=pd.read_csv("homeprices.csv")
df


# In[34]:


dummies=pd.get_dummies(df.town)
dummies


# In[21]:


merged=pd.concat([df,dummies],axis='columns')
merged


# In[22]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()


# In[30]:


x=df.drop('price',axis='columns')
x


# In[27]:


y=df.price
y


# In[31]:


model.fit(x,y)


# In[32]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[35]:


dfle=df
le.fit_transform(dfle.town)


# In[36]:


import pandas as pd
df = pd.read_csv("homeprices.csv")


# In[37]:


dummies = pd.get_dummies(df.town)
dummies


# In[39]:


import pandas as pd


# In[40]:


df=pd.read_csv('carprices.csv')


# In[41]:


df


# In[46]:


dummies=pd.get_dummies(df['Car Model'])
dummies


# In[47]:


merged=pd.concat([df,dummies],axis='columns')
merged


# In[51]:


final=merged.drop(['Car Model',"Mercedez Benz C class"],axis='columns')
final


# In[52]:


x=final.drop('Sell Price($)',axis='columns')
x


# In[53]:


y=final['Sell Price($)']


# In[54]:


y


# In[55]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()


# In[56]:


model.fit(x,y)


# In[57]:


model.score(x,y)


# In[59]:


model.predict([[86000,7,0,1]])


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits


# In[3]:


digits=load_digits()


# In[4]:


dir(digits)


# In[5]:


digits.data[0]


# In[10]:


plt.gray()
for i in range(5):
    plt.matshow(digits.images[i])


# In[12]:


digits.target[0:5]


# In[13]:


from sklearn.model_selection import train_test_split


# In[15]:


X_train, X_test ,y_train, y_test= train_test_split(digits.data, digits.target,test_size=0.2)


# In[16]:


len(X_train)


# In[17]:


len(X_test)


# In[18]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()


# In[19]:


model.fit(X_train,y_train)


# In[20]:


model.score(X_test,y_test)


# In[22]:


plt.matshow(digits.images[67])


# In[24]:


digits.target[67]


# In[26]:


model.predict([digits.data[67]])


# In[28]:


model.predict(digits.data[0:5])


# In[30]:


y_predicted=model.predict(X_test)
from sklearn.metrics import confusion_matrix

cn=confusion_matrix(y_test,y_predicted)
cn


# In[32]:


import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cn,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# Decision Tree Machine Learning

# In[34]:


import pandas as pd


# In[35]:


df=pd.read_csv("salaries.csv")
df.head()


# In[36]:


inputs=df.drop('salary_more_then_100k',axis='columns')


# In[38]:


target=df['salary_more_then_100k']


# In[39]:


from sklearn.preprocessing import LabelEncoder
le_company=LabelEncoder()
le_job=LabelEncoder()
le_degree=LabelEncoder()


# In[40]:


inputs['company_n']=le_company.fit_transform(inputs['company'])
inputs['job_n']=le_job.fit_transform(inputs['job'])
inputs['degree_n']=le_degree.fit_transform(inputs['degree'])


# In[41]:


inputs


# In[45]:


inputs_n=inputs.drop(['company','job','degree'],axis='columns')


# In[46]:


inputs_n


# In[42]:


target


# In[43]:


from sklearn import tree
model=tree.DecisionTreeClassifier()


# In[47]:


model.fit(inputs_n,target)


# In[48]:


model.score(inputs_n,target)


# Is salary of Google, Computer Engineer, Bachelors degree>100k?

# In[49]:


model.predict([[2,1,0]])

Is salary of Google, Computer Engineer, Bachelors degree>100k?
# In[50]:


model.predict([[2,1,1]])


# # Support Vector Machine

# In[51]:


import pandas as pd
from sklearn.datasets import load_iris
iris=load_iris()


# In[53]:


iris.feature_names


# In[54]:


iris.target_names


# In[55]:


df=pd.DataFrame(iris.data,columns=iris.feature_names)
df.head()


# In[56]:


df['target']=iris.target
df.head()


# In[57]:


df[df.target==1].head()


# In[58]:


df[df.target==2].head()


# In[59]:


df['flower_name']=df.target.apply(lambda x:iris.target_names[x])
df.head()


# In[60]:


df[45:55]


# In[61]:


df0=df[:50]
df1=df[50:100]
df2=df[100:]


# In[62]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Sepal length vs Petal Length (Setosa vs Versicolor)

# In[63]:


plt.xlabel('Sepal length')
plt.ylabel('Sepal Width')
plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],color="green",marker="+")
plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],color="blue",marker=".")


# Train using SVM

# In[64]:


from sklearn.model_selection import train_test_split


# In[65]:


X=df.drop(['target','flower_name'],axis='columns')
y=df.target


# In[66]:


X_train,X_test, y_tarin, y_test=train_test_split(X,y,test_size=0.2)


# In[67]:


len(X_train)


# In[68]:


len(X_test)


# In[70]:


from sklearn.svm import SVC
model=SVC()


# In[71]:


model.fit(X_train,y_train)


# In[4]:


import pandas as pd
from sklearn.datasets import load_digits
digits=load_digits()


# In[5]:


dir(digits)


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.gray()
for i in range (4):
    plt.matshow(digits.images[i])


# In[9]:


df=pd.DataFrame(digits.data)


# In[10]:


digits.data[:5]


# In[12]:


df['target']=digits.target
df.head()


# In[14]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df.drop(['target'],axis='columns'),digits.target,test_size=0.2)


# In[15]:


len(X_test)


# In[16]:


from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(X_train,y_train)


# In[17]:


model.score(X_test,y_test)


# In[ ]:





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






