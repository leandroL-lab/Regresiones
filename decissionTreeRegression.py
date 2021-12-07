#!/usr/bin/env python
# coding: utf-8

# In[2]:


#REGRESION CON ARBOLES DE DECISION

import numpy as np
import matplotlib.pyplot as plt
#Importar el data set
import pandas as pd
df = pd.read_csv('Position_Salaries.csv')
X = df.iloc[:,1:2].values
y = df.iloc[:,2].values


# In[48]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1,1))


# In[6]:


from sklearn.tree import DecisionTreeRegressor

regression = DecisionTreeRegressor(random_state = 0)
regression.fit(X,y)


# In[ ]:





# In[7]:


y_pred = regression.predict([[6.5]])


# In[8]:


y_pred


# In[25]:


# Visualización de los resultados del Decision Tree

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(-1, 1)

plt.scatter(X, y, color = "red")
plt.plot(X_grid,regression.predict(X_grid), color = "black")
plt.title("Modelo de Regresión Decision Tree  \n")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()


# In[ ]:




