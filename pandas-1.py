
# coding: utf-8

# In[1]:


#creating an empty panel
import pandas as pd
import numpy as np
data = np.random.rand(2,4,5)
p = pd.Panel(data)
p


# In[6]:


data = {'a':pd.DataFrame(np.random.rand(4,3)), 'b':pd.DataFrame(np.random.rand(4,2))}
p = pd.Panel(data)
print(p['a'])
print(p.major_xs(1))
print(p.minor_xs(1))


# In[7]:


#create a series with 100 random numbers 
import pandas as pd
import numpy as np
p = pd.Series(np.random.rand(4))
p


# In[8]:


print('print the axes are:')
print(p.axes)


# In[9]:


print('print the whether the object is empty or not:')
print(p.empty)


# In[11]:


print('print the diemnsions of the object')
print(p.ndim)


# In[12]:


print('prin the size of the object:')
print(p.size)


# In[13]:


print('return the actual data series is:')
print(p.values)


# In[15]:


import pandas as pd
import numpy as np
p = pd.Series(np.random.rand(4))
print(p)
print('print the first rows of the series')
print(p.head(2))
print('return the last rows of the series')
print(p.tail(2))


# In[21]:


#dataframe basic functionalities
import pandas as pd
import numpy as np
data = {'Names':pd.Series(['mahesh','rajesh','unkush','rukush','chankush']), 'age':pd.Series([42,41,35,63,25]), 
        'rating':pd.Series([42.1,21.0,36.0,12.4,42.1])}
p = pd.DataFrame(data)
print(p)


# In[22]:


#Transpose
print(p.T)


# In[23]:


#axes
print(p.axes)


# In[24]:


#dtypes
print(p.dtypes)


# In[25]:


#checking whether the dataframe is empty or not
print(p.empty)


# In[26]:


#ndim
print(p.ndim)


# In[27]:


#shape
print(p.shape)


# In[28]:


#size
print(p.size)


# In[29]:


#values
print(p.values)


# In[30]:


#head
print(p.head(2))


# In[31]:


print(p.tail(2))


# In[33]:


import pandas as pd
import numpy as np
data = {'names':pd.Series(['mahesh','rajesh','ramesh','raju','rinku','pinku','janku','danku']), 
        'age':pd.Series([41,42,21,25,54,36,24,82])}
p = pd.DataFrame(data)
print(p)
#sum()
print(p.sum())


# In[37]:


#axis
print(p.sum(1))


# In[38]:


#mean
print(p.mean())


# In[39]:


#std
print(p.std())


# In[40]:


#describe
print(p.describe())


# In[41]:


#include object
print(p.describe(include=['object']))


# In[43]:


print(p.describe(include='all'))


# In[49]:


import pandas as pd
import numpy as np
def adder(e1e1,e1e2):
    return e1e1+e1e2
    
df = pd.DataFrame(np.random.randn(5,3), columns=['col1','col2','col3'])
df.pipe(adder, 2)
df


# In[50]:


print(df.apply(np.mean))


# In[54]:


print(df.apply(np.mean,axis=1))


# In[58]:


import pandas as pd
import numpy as np
df = pd.DataFrame(np.random.randn(5,3), columns=['col1','col2','col3'])
print(df)
print(df.apply(lambda x:x.max()-x.min()))

print(df.apply(np.mean))


# In[60]:


print(df['col1'].map(lambda x:x*100))
print(df.apply(np.mean))


# In[61]:


#reindexing
import pandas as pd
import numpy as np

N = 20
df = pd.DataFrame({'A':pd.date_range(start='2016-01-01',periods=N, freq='D'), 'x':np.linspace(0, stop=N-1, num=N),
                  'y':np.random.rand(N), 'c':np.random.choice(['low','medium','high'],N).tolist(),
                  'D':np.random.normal(100,10, size=(N)).tolist()})
df.reindexed = df.reindex(index=[0,2,5], columns=['A','C','D'])
print(df.reindexed)


# In[64]:


import pandas as pd
import numpy as np
df = pd.DataFrame(np.random.randn(10,3), columns=['col1','col2','col3'])
df1 = pd.DataFrame(np.random.randn(7,3), columns=['col1','col2','col3'])

df = df.reindex_like(df1)
df


# In[76]:


import pandas as pd
import numpy as np
df = pd.DataFrame(np.random.randn(6,3), columns=['col1','col2','col3'])
df1 = pd.DataFrame(np.random.randn(2,3), columns=['col1','col2','col3'])
print(df1.reindex_like(df))


#forward filling
print(df1.reindex_like(df, method='ffill'))
#limits while filling
print(df1.reindex_like(df, method='ffill', limit=1))


# In[77]:


#renaming the cols
import pandas as pd
import numpy as np
df = pd.DataFrame(np.random.randn(4,3), columns=['col1','col2','col3'])
print(df)
print('after renaming the cols:')
print(df.rename(columns={'col1':'c1', 'col2':'c2','col3':'c3'}))

