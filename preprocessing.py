#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import matplotlib.pyplot as plt
#Importar el data set
import pandas as pd
df = pd.read_csv('Data.csv')
X = df.iloc[:,:-1].values
y = df.iloc[:,3].values


# In[15]:


#Tratamiento de NAs
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = "mean", verbose=0)
imputer = imputer.fit(X[:,1:3]) 
X[:, 1:3] = imputer.transform(X[:,1:3])


# In[23]:


# Codificar datos categoricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],   
    remainder='passthrough'                        
)

X = np.array(ct.fit_transform(X), dtype=np.float)
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


# In[39]:


#Dividir los datros en conjuntos de test y train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)


# In[40]:


#Escalado de variables 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# In[41]:


X_train


# In[ ]:




