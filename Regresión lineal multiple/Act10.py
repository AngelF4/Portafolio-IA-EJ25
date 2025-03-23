import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Suponiendo que ya tienes cargado el DataFrame `filtered_data`

# Creamos una nueva variable que es la suma de:
# enlaces, comentarios (rellenando nulos con 0), e imágenes
suma = (
    filtered_data["# of Links"] +
    filtered_data['# of comments'].fillna(0) +
    filtered_data['# Images video']
)

# Creamos el DataFrame de variables independientes
dataX2 = pd.DataFrame()
dataX2["Word count"] = filtered_data["Word count"]
dataX2["suma"] = suma

# Variables de entrada (X) y de salida (Z)
XY_train = np.array(dataX2)
z_train = filtered_data['# Shares'].values

# Creamos el modelo de regresión lineal múltiple
regr2 = linear_model.LinearRegression()

# Entrenamos el modelo
regr2.fit(XY_train, z_train)

# Realizamos predicciones
z_pred = regr2.predict(XY_train)

# Resultados
print("Coefficients:", regr2.coef_)
print("Independent term (intercept):", regr2.intercept_)
print("Mean squared error:", mean_squared_error(z_train, z_pred))
print("Variance score (R²):", r2_score(z_train, z_pred))