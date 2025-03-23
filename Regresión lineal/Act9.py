# Importar librerías necesarias
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Cargar los datos desde la URL del dataset
url = "https://raw.githubusercontent.com/jbagnato/machine-learning/master/datasets/articles.csv"
data = pd.read_csv(url)

# Filtrar datos: artículos con menos de 3000 palabras y menos de 80,000 compartidos
filtered_data = data[(data["Word count"] < 3000) & (data["# Shares"] < 80000)]

# Crear nueva variable "suma" (interacciones): enlaces + comentarios + imágenes/video
suma = (filtered_data["# of Links"] +
        filtered_data["# of comments"].fillna(0) +
        filtered_data["# Images video"])

# Crear DataFrame con las variables predictoras
dataX2 = pd.DataFrame()
dataX2["Word count"] = filtered_data["Word count"]
dataX2["suma"] = suma

# Variable de salida
z_train = filtered_data["# Shares"].values

# Matriz de entrada
XY_train = np.array(dataX2)

# Crear y entrenar el modelo de regresión lineal múltiple
regr2 = linear_model.LinearRegression()
regr2.fit(XY_train, z_train)

# Predecir sobre los datos de entrenamiento
z_pred = regr2.predict(XY_train)

# Mostrar resultados
print("Coeficientes:", regr2.coef_)
print("Intercepto:", regr2.intercept_)
print("Error cuadrático medio (MSE):", mean_squared_error(z_train, z_pred))
print("Coeficiente de determinación (R²):", r2_score(z_train, z_pred))

# Hacer una predicción para un artículo con:
# 2000 palabras, 10 enlaces, 4 comentarios, 6 imágenes
interaccion = 10 + 4 + 6
prediccion = regr2.predict([[2000, interaccion]])
print("Predicción para un artículo con 2000 palabras y 20 elementos de interacción:", int(prediccion[0]))