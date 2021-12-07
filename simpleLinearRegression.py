#!/usr/bin/env python
# coding: utf-8

# In[92]:


#Regresion lineal simple
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('Salary_Data.csv')
X = df.iloc[:,:-1].values
y = df.iloc[:,1].values


# In[93]:


#Dividir los datros en conjuntos de test y train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 1/3, random_state = 0)


# In[94]:


#Model of simple linear regression
from sklearn.linear_model import LinearRegression

regression = LinearRegression()
regression.fit(X_train, y_train)

#Predecir el conjunto de test
X_pred = regression.predict(X_test)


# In[98]:


#Visualiacion de resultaods de entrenamiento
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regression.predict(X_train), color = 'blue')
plt.title('Sueldo vs Years de experiencia [Conjunto de entrenamiento]')
plt.xlabel("Years de experiencia")
plt.ylabel('Sueldo')
plt.show()


# In[100]:


#Visualiacion de resultaods de test
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regression.predict(X_train), color = 'blue')
plt.title('Sueldo vs Years de experiencia [Conjunto de Testing]')
plt.xlabel("Years de experiencia")
plt.ylabel('Sueldo')
plt.show()

