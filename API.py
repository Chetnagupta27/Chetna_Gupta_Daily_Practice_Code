#!/usr/bin/env python
# coding: utf-8

# In[5]:


import requests
response=requests.get('https://codingninjas.in/api/v3/courses')
print(response.status_code)
print(response.encoding)
#print(response.text)
print(response.url)
print(response.headers)
print()
print(type(response.headers)) ##it is case sensetive
header_info=response.headers
print(header_info['Date'])
print(header_info['Content-encoding'])
print(header_info['Content-Type'])


# In[15]:


test=requests.get(' http://api.open-notify.org/iss-pass')
print(test.status_code)

#Here the output is 404 because it is wrong API, no any API exist with this URL


# In[16]:


test2=requests.get('http://api.open-notify.org/iss-now.json')
print(test2.status_code)
#Here the output is 200 bcoz it is an accessible API


# In[6]:


import json
json_data='{"Student" : " Mohit"}'
python_data=json.loads(json_data)
print(type(python_data))
print(python_data['Student'])


# In[23]:


json_data='{"roll_no":null}'
python_data=json.loads(json_data)
print(python_data['roll_no'])
print(type(python_data['roll_no']))


# In[25]:


json_data='{"Student":{"Name" : " Mohit","Roll_no" : 100}}'
python_data=json.loads(json_data)
student_details=python_data['Student']
print(type(student_details["Name"]))


# In[8]:


import json
python_data=json.loads(response.text)
python_data


# In[9]:


type(python_data)


# In[11]:


all_courses=python_data['data']["courses"]
all_courses


# In[30]:


for i in all_courses:
    #if all_courses['available_offline']=="False":
    print(i['title'])
print()   
print(len(all_courses))


# In[22]:


p= requests.get('https://dog.ceo/api/breeds/list/all')
#print(p.text)
python_data=p.json()
print(type(python_data))
#print(python_data)
for key,value in python_data['message'].items():
    print(key,",",len(value))


# In[24]:


for i in python_data['message']['spaniel']:
    print(i)


# In[43]:


import json
import requests
data=requests.get('https://api.codingninjas.com/api/v3/courses')
python_data=data.json()


# In[44]:


all_course=python_data['data']['courses']
for i in all_course:
    if i["available_online"]=="true" and i["available_offline"]=="false":
        print(i["title"])


# In[46]:


import json
import requests
data=requests.get('https://api.codingninjas.com/api/v3/courses')
python_data=data.json()
all_course=python_data['data']['courses']
for i in all_course:
    if i["available_online"]=="true" and i["available_offline"]=="false":
        print(i["title"])


# In[53]:


import json
import requests
data2=requests.get('https://api.codingninjas.com/api/v3/events?event_category=ALL_EVENTS&event_sub_category=All%20Time%20Favorites&tag_list=&offset=0&_ga=2.245846535.2050912122.1663326663-910016661.1656695624')
python_data=data2.json()
events=python_data['data']["events"]
print(events[0]['short_desc'])


# In[60]:


import json
import requests
a=requests.get('http://api.codingninjas.com/api/v3/courses', params={"id":"19"})
print(a.json())
#print(python_data)
#for i in python_data['data']['courses']:
    #print(i["title"])
    


# In[65]:


import json
import requests
data=requests.get('https://api.openaq.org/docs#/v2/citiesgetv2citiesget',params={"country":"AU"})
data.status_code
data


# In[ ]:




