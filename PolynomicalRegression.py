#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
#Importar el data set
import pandas as pd
df = pd.read_csv('Position_Salaries.csv')
X = df.iloc[:,1:2].values
y = df.iloc[:,2].values


# In[9]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X,y)


# In[24]:


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)


# In[25]:


#Visualizacion de los resultados del Modelo Lineal
plt.scatter(X,y,color = 'red')
plt.plot(X,lin_reg.predict(X), color = 'blue')
plt.title("Modelo de Regresion Lineal")
plt.xlabel("Posicion del Empleado")
plt.ylabel("Sueldo")
plt.show()


# In[26]:


#Visualizacion de los resultados del Modelo Polinomico
X_grid = np.arange(min(X),max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X,y,color = 'red')
plt.plot(X_grid,lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title("Modelo de Regresion Polinomica")
plt.xlabel("Posicion del Empleado")
plt.ylabel("Sueldo")
plt.show()


# In[29]:


#Prediccion de nuestros modelos

lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))


# In[ ]:




