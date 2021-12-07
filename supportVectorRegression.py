#!/usr/bin/env python
# coding: utf-8

# In[47]:


#SVR
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


# In[49]:


from sklearn.svm import SVR

regression = SVR(kernel = "rbf")
regression.fit(X,y.ravel())


# In[50]:


y_pred = regression.predict(sc_X.transform([[6.5]]))
y_pred = sc_y.inverse_transform(y_pred)


# In[51]:


y_pred


# In[53]:


# Visualización de los resultados del SVR (con valores intermedios y sin las variables escaladas)
X_2 = sc_X.inverse_transform(X)
y_2 = sc_y.inverse_transform(y)
X_grid = np.arange(min(X_2), max(X_2), 0.1)
X_grid = X_grid.reshape(-1, 1)
plt.scatter(X_2, y_2, color = "red")
plt.plot(X_grid, sc_y.inverse_transform(regression.predict(sc_X.transform(X_grid))), color = "blue")
plt.title("Modelo de Regresión (SVR) (con valores intermedios y sin las variables escaladas)  \n")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()


# In[ ]:




