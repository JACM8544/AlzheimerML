import pandas as pd

# Ruta del archivo de entrenamiento
train_data_path = r"C:\Users\Technologic PC\PycharmProjects\AlzheimerML\Alzheimer\Entrenamiento\4image_stats.csv"

# Ruta del archivo de prueba (test)
test_data_path = r"C:\Users\Technologic PC\PycharmProjects\AlzheimerML\Alzheimer\Implementacion\test_image_stats.csv"

# Cargar ambos archivos
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

# Verificar columnas
print("Columnas del archivo de entrenamiento:", train_data.columns)
print("Columnas del archivo de prueba:", test_data.columns)

# Reindexar las columnas del archivo de prueba para que coincidan con las del entrenamiento
test_data = test_data.reindex(columns=train_data.columns)

# Verificar que las columnas ahora coincidan
print("Columnas después de reindexar (prueba):", test_data.columns)
