#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Regresion lineal simple
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('50_Startups.csv')
X = df.iloc[:,:-1].values
y = df.iloc[:,4].values


# In[5]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X = LabelEncoder()

X[:, 3] = labelencoder_X.fit_transform(X[:, 3])

onehotencoder = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [3])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                         # Leave the rest of the columns untouched
)

X = onehotencoder.fit_transform(X)

#evitar la multiple colinealidad de las variables dummy
X = X[:,1:]


# In[7]:


#Dividir los datros en conjuntos de test y train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)


# In[9]:


#Modelo de regresion lineal simple
from sklearn.linear_model import LinearRegression

regression = LinearRegression()
regression.fit(X_train, y_train)

#Predecir el conjunto de test
y_pred = regression.predict(X_test)


# In[11]:


X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)


# In[17]:


#Construir el modelo optimo de RLM utilizando le Eliminacion hacia atras
import statsmodels.api as sm

SL = 0.05

X_opt = X[:,[0,1,2,3,4,5]].tolist()
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regression_OLS.summary()


# In[18]:


X_opt = X[:,[0,1,3,4,5]].tolist()
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regression_OLS.summary()


# In[19]:


X_opt = X[:,[0,3,4,5]].tolist()
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regression_OLS.summary()


# In[20]:

#La columna 5 presentaba un p-value cercano al 0.5 fue eliminado para seguir el modelo de eliminacion hacia atras
#pero daba para una evaluacion mas detallada con su r cuadrado, cosa que se hace en la funcion siguiente
X_opt = X[:,[0,3,5]].tolist()
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regression_OLS.summary()


# In[24]:

#El modelo MLR termino siendo una regresion lineal simple por la elminacion de la columna 5
X_opt = X[:,[0,3]].tolist()
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regression_OLS.summary()


# In[25]:

#Corroboramos si la elminiacion de  la columna 5 fue correcta con esta funcion
def backwardElimination(x, SL):    
    numVars = len(x[0])    
    temp = np.zeros((50,6)).astype(int)    
    for i in range(0, numVars):        
        regressor_OLS = sm.OLS(y, x.tolist()).fit()        
        maxVar = max(regressor_OLS.pvalues).astype(float)        
        adjR_before = regressor_OLS.rsquared_adj.astype(float)        
        if maxVar > SL:            
            for j in range(0, numVars - i):                
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
                    temp[:,j] = x[:, j]                    
                    x = np.delete(x, j, 1)                    
                    tmp_regressor = sm.OLS(y, x.tolist()).fit()                    
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)                    
                    if (adjR_before >= adjR_after):                        
                        x_rollback = np.hstack((x, temp[:,[0,j]]))                        
                        x_rollback = np.delete(x_rollback, j, 1)     
                        print (regressor_OLS.summary())                        
                        return x_rollback                    
                    else:                        
                        continue    
    regressor_OLS.summary()    
    return x 
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]

#Nuestro modelo decide conservar la columna 5 y crear un verdadero MLR
X_Modeled = backwardElimination(X_opt, SL)


# In[ ]:




