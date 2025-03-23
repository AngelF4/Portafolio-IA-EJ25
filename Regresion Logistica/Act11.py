# Importar librerías necesarias
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report

# Cargar los datos
url = "https://raw.githubusercontent.com/jbagnato/machine-learning/master/datasets/usuarios_win_mac_lin.csv"
data = pd.read_csv(url)

# Exploración inicial
print(data.head())
print(data.describe())
print(data.groupby('clase').size())

# Visualización
sns.pairplot(data, hue='clase')
plt.show()

# Preparar datos
X = data.drop(columns='clase')
y = data['clase']

# Dividir en conjunto de entrenamiento y validación
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)

# Evaluación del modelo
y_pred = model.predict(X_test)

print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# Validación cruzada
scores = cross_val_score(model, X, y, cv=5)
print(f"\nPrecisión promedio (cross-validation): {scores.mean():.2f}")

# Predicción de nuevos datos
# Ejemplo: duración = 3.5, páginas vistas = 7, acciones = 2, valor = 35.0
nuevo_usuario = np.array([[3.5, 7, 2, 35.0]])
prediccion = model.predict(nuevo_usuario)
print(f"\nSistema operativo predicho para nuevo usuario: {prediccion[0]}")