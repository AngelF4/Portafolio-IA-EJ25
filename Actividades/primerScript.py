import os
import pandas as pd
import re
import kagglehub

# =============================================================================
# 1. DESCARGA DEL DATASET MEDIANTE KAGGLEHUB
# =============================================================================

# Descarga la última versión del dataset. Esto creará una carpeta con los archivos descargados.
path = kagglehub.dataset_download("vikrishnan/iris-dataset")
print("Path to dataset files:", path)

# =============================================================================
# 2. CARGA DEL DATASET
# =============================================================================

# Se asume que el archivo CSV se llama "Iris.csv" y se encuentra directamente en el directorio indicado por 'path'
csv_file = os.path.join(path, "Iris.csv")
try:
    df = pd.read_csv(csv_file)
    print("Dataset cargado exitosamente.")
except Exception as e:
    print("Error al cargar el CSV:", e)
    exit()

# =============================================================================
# 3. EXPLORACIÓN PRELIMINAR
# =============================================================================

# a) Dimensiones del DataFrame
print("\nDimensiones del DataFrame:", df.shape)

# b) Lista de columnas y sus tipos de datos
print("\nTipos de variables:")
print(df.dtypes)

# c) Primeros registros del dataset
print("\nPrimeros 5 registros:")
print(df.head())

# d) Estadísticas descriptivas (para columnas numéricas)
print("\nEstadísticas descriptivas:")
print(df.describe())

# =============================================================================
# 4. VALIDACIONES BÁSICAS DEL DATASET
# =============================================================================

# Se asume que el dataset consta de 5 columnas:
# 4 columnas numéricas: sepel_length, sepal_width, petal_length, petal_width
# 1 columna categórica: species
# Es posible que los nombres de las columnas varíen ligeramente. Se intenta normalizar renombrándolas:
columnas_esperadas = {
    "sepal_length": ["sepal length", "sepal_length"],
    "sepal_width": ["sepal width", "sepal_width"],
    "petal_length": ["petal length", "petal_length"],
    "petal_width": ["petal width", "petal_width"],
    "species": ["species"]
}

# Función para mapear nombres de columnas
def normalizar_nombre(nombre):
    return nombre.strip().lower().replace(" ", "_")

columnas_existentes = { normalizar_nombre(col): col for col in df.columns }

# Renombrar las columnas a los nombres esperados, si son detectados
nuevos_nombres = {}
for clave, posibles in columnas_esperadas.items():
    for posible in posibles:
        if posible in columnas_existentes:
            nuevos_nombres[columnas_existentes[posible]] = clave
            break

if nuevos_nombres:
    df.rename(columns=nuevos_nombres, inplace=True)
    print("\nColumnas renombradas a:")
    print(df.columns)
else:
    print("\nNo se realizó renombramiento de columnas; se usarán los nombres originales.")

# -----------------------------------------------------------------------------
# 4.1 Validación de columnas numéricas
# -----------------------------------------------------------------------------

# Lista de columnas numéricas esperadas
numericas = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

def validar_valor_positivo(valor):
    """Verifica que el valor numérico sea mayor o igual a cero (y no nulo)."""
    if pd.isna(valor):
        return False
    return valor >= 0

for col in numericas:
    if col not in df.columns:
        print(f"[ADVERTENCIA] La columna numérica '{col}' no se encontró en el dataset.")
        continue
    # Aplicamos la validación a cada elemento de la columna
    invalidos = df[~df[col].apply(validar_valor_positivo)]
    if not invalidos.empty:
        print(f"\n[ADVERTENCIA] Columna '{col}' contiene valores negativos o nulos:")
        print(invalidos[[col]])
    else:
        print(f"\n[OK] La columna '{col}' pasó la validación de valores positivos.")

# -----------------------------------------------------------------------------
# 4.2 Validación de la columna categórica 'species'
# -----------------------------------------------------------------------------

# Se esperan tres especies: setosa, versicolor y virginica (posiblemente con o sin prefijo "iris-")
patron_especies = re.compile(r"^(iris-)?(setosa|versicolor|virginica)$", re.IGNORECASE)

def validar_especie(valor):
    """Verifica que el valor en 'species' corresponda a una de las especies esperadas."""
    if pd.isna(valor):
        return False
    return bool(patron_especies.match(valor.strip()))

if 'species' in df.columns:
    invalid_species = df[~df['species'].apply(validar_especie)]
    if not invalid_species.empty:
        print("\n[ADVERTENCIA] Se encontraron especies fuera del patrón esperado:")
        print(invalid_species[['species']])
    else:
        print("\n[OK] La columna 'species' pasó la validación.")
else:
    print("\n[ADVERTENCIA] La columna 'species' no se encontró en el dataset.")