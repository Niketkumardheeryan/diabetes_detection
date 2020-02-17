
# coding: utf-8

# In[32]:


import numpy as np
import pandas as pd

#collecting data 
data=pd.read_csv('diabetes2.csv')
#data.isna().sum()

#dividing data into dependent or independent variables

x=data.iloc[:, :-1].values
y=data.iloc[:,8].values

#no need of scaling and encoding here

#spliting dataset 


from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)



#fitting models

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=0)
classifier.fit(x_train,y_train)

prediction=classifier.predict(x_test)

prediction





# In[26]:


data.head()

