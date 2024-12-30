import pandas as pd
import joblib
from sklearn.metrics import classification_report

# Rutas de los archivos
train_data_path = r"C:\Users\Technologic PC\PycharmProjects\AlzheimerML\Alzheimer\Entrenamiento\4image_stats.csv"
test_data_path = r"C:\Users\Technologic PC\PycharmProjects\AlzheimerML\Alzheimer\Implementacion\test_image_stats.csv"
scaler_path = r"C:\Users\Technologic PC\PycharmProjects\AlzheimerML\Alzheimer\Entrenamiento\scaler.pkl"
encoder_path = r"C:\Users\Technologic PC\PycharmProjects\AlzheimerML\Alzheimer\Entrenamiento\encoder.pkl"
model_path = r"C:\Users\Technologic PC\PycharmProjects\AlzheimerML\Alzheimer\Entrenamiento\svm_model.pkl"

# Cargar los datos
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

# Reindexar columnas del archivo de prueba para que coincidan con las del entrenamiento
test_data = test_data.reindex(columns=train_data.columns)

# Separar características y etiquetas
X_test = test_data.drop(columns=["label"])
y_test = test_data["label"]

# Normalizar las características del conjunto de prueba
scaler = joblib.load(scaler_path)
X_test_scaled = scaler.transform(X_test)

# Codificar las etiquetas del conjunto de prueba
encoder = joblib.load(encoder_path)
y_test_encoded = encoder.transform(y_test)

# Cargar el modelo entrenado
model = joblib.load(model_path)

# Realizar predicciones
y_test_pred = model.predict(X_test_scaled)

# Generar reporte de clasificación
print("Reporte de Clasificación para Implementación:")
print(classification_report(y_test_encoded, y_test_pred))
