
# coding: utf-8

# In[2]:


import pandas as pd
a = pd.Series()
a


# In[4]:


import pandas as pd
import numpy as np
data = np.array([1,2,3,4,5,6])
a = pd.Series(data)
print(a)imp


# In[8]:


import pandas as pd
import numpy as np
data = np.array(['a','b','c','d'])
a = pd.Series(data, index=[100,101,102,103])
a


# In[10]:


import pandas as pd
import numpy as np
data = {'a':0.,'b':1.,'c':2.,'d':3.,'e':4.}
a = pd.Series(data)
a


# In[11]:


data = {'a':1,'b':2,'c':3,'d':4,'e':5}
a = pd.Series(data, index=['b','e','f','a','c','d'])
a


# In[12]:


a = pd.Series(5, index=[1,2,3,4,5,6])
a


# In[16]:


#Accessing Data from Series with Position
a = pd.Series([1,2,3,4,5,6], index=['a','b','c','d','e','f'])
a['a']


# In[17]:


a[:3]


# In[18]:


a[-3:]


# In[26]:


a = pd.Series([1,2,3,4,5], index=['a','b','c','d','e'])
print(a['a'])
print(a['b'])
print(a['c'])
print(a['d'])
print(a['e'])
print(a[['a','b','c','d','e']])


# In[28]:


#creating an empty dataframe
import pandas as pd
df = pd.DataFrame()
df


# In[29]:


data = [1,2,3,4,5,6]
a = pd.DataFrame(data)
a


# In[30]:


data = [['Alex',23],['mahesh',43],['sundar',44],['ramesh',54]]
a = pd.DataFrame(data, columns=['Name','age'])
a


# In[31]:


data = [['mahesh',43],['rajesh',42],['rankush',41],['eshwar',52]]
a = pd.DataFrame(data, columns=['Names','age'], dtype=float)
a


# In[33]:


#creating a dataFrame from a dict of ndarray/lists
data = {'Name':['mahesh','rajesh','enkush','rinkush'], 'age':[42,15,63,78]}
a = pd.DataFrame(data)
a


# In[34]:


data = {'Name':['mahesh','rajesh','enkush','rinkush'], 'age':[42,15,63,78]}
a = pd.DataFrame(data, index=['rank1','rank2','rank3','rank4'])
a


# In[35]:


#create a dataframe from a list of dicts
data = [{'a':1,'b':2,'c':3,'d':4,'e':5,'f':6},{'g':7,'h':8}]
a = pd.DataFrame(data)
a


# In[37]:


data =[{'a':1,'b':2},{'a':1,'b':3,'c':4}]
a = pd.DataFrame(data, index=['first','second'])
a


# In[39]:


data =[{'a':1,'b':2},{'a':1,'b':3,'c':4}]
a = pd.DataFrame(data, index=['first','second'], columns=['a','b'])
a1 = pd.DataFrame(data, index=['first','second'], columns=['a','b1'])
print(a)
print(a1)


# In[45]:


#create a dataframe from dict of series
data = {'a': pd.Series([1,2,3], index=['a','b','c']), 'b':pd.Series([1,2,3,4], index=['a','b','c','d'])}
a = pd.DataFrame(data)
a


# In[46]:


data = {'a': pd.Series([1,2,3], index=['a','b','c']), 'b':pd.Series([1,2,3,4], index=['a','b','c','d'])}
a = pd.DataFrame(data)
a['a']


# In[61]:


data = {'a': pd.Series([1,2,3], index=['a','b','c']), 'b':pd.Series([1,2,3,4], index=['a','b','c','d'])}
df = pd.DataFrame(data)
#adding a new column
df['c']=pd.Series([10,20,30], index=['a','b','c'])
print(df)
#creating a column with existing col
df['d'] = a['a']+a['b']
print(df)



# In[62]:


#column deletion
print('deletion of the dataframe column')
del df['a']
print(df)
print('deletion of another column by using pop()')
df.pop('b')
print(df)


# In[63]:


#row selection,deletion,addition
#rows can be selected by passing row label to a loc function
d = {'one': pd.Series([1,2,3], index=['a','b','c']), 'two':pd.Series([1,2,3,4], index=['a','b','c','d'])}
df = pd.DataFrame(d)
print(df.loc['b'])


# In[64]:


#row selection by integer location
print(df.iloc[2])


# In[66]:


#slicing
print(df[2:4])
print(df[-3:])


# In[70]:


#addition of rows
df = pd.DataFrame([[1,2],[3,4]], columns=['a','b'])
df1 = pd.DataFrame([[4,5],[6,7]], columns=['a','b'])
df = df.append(df1)
df


# In[71]:


#deletion of rows
df = df.drop(0)
df

