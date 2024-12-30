from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

# Ruta del CSV con las estadísticas
input_path = r"C:\Users\Technologic PC\PycharmProjects\AlzheimerML\Alzheimer\Entrenamiento\4image_stats.csv"

# Cargar los datos
data = pd.read_csv(input_path)

# Separar características y etiquetas
X = data.drop(columns=["label"])  # Características: todas las columnas excepto 'label'
y = data["label"]  # Etiquetas

# Normalizar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Codificar las etiquetas en valores numéricos
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Mostrar datos procesados
print("Características normalizadas (primeras 5 filas):")
print(X_scaled[:5])

print("Etiquetas codificadas (primeras 5):")
print(y_encoded[:5])

# Guardar el escalador y el codificador para uso futuro
import joblib
scaler_path = r"C:\Users\Technologic PC\PycharmProjects\AlzheimerML\Alzheimer\Entrenamiento\scaler.pkl"
encoder_path = r"C:\Users\Technologic PC\PycharmProjects\AlzheimerML\Alzheimer\Entrenamiento\encoder.pkl"
joblib.dump(scaler, scaler_path)
joblib.dump(encoder, encoder_path)

print(f"Escalador guardado en: {scaler_path}")
print(f"Codificador guardado en: {encoder_path}")

from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Crear y entrenar un modelo SVM
svm_model = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
svm_model.fit(X_scaled, y_encoded)

# Hacer predicciones (usaremos los mismos datos de entrenamiento para evaluación inicial)
y_pred = svm_model.predict(X_scaled)

# Evaluar el modelo
print("Reporte de Clasificación:")
print(classification_report(y_encoded, y_pred))

print(f"Precisión del modelo: {accuracy_score(y_encoded, y_pred):.2f}")

# Cargar los datos de prueba
test_path = r"C:\Users\Technologic PC\PycharmProjects\AlzheimerML\Alzheimer\Entrenamiento\4image_stats.csv"
test_data = pd.read_csv(test_path)

# Preparar características y etiquetas
X_test = test_data.drop(columns=["label"])
y_test = test_data["label"]

# Normalizar usando el escalador guardado
scaler = joblib.load(scaler_path)
X_test_scaled = scaler.transform(X_test)

# Codificar etiquetas
encoder = joblib.load(encoder_path)
y_test_encoded = encoder.transform(y_test)

# Hacer predicciones y evaluar
y_test_pred = svm_model.predict(X_test_scaled)
print("Reporte de Clasificación en Datos de Prueba:")
print(classification_report(y_test_encoded, y_test_pred))

model_path = r"C:\Users\Technologic PC\PycharmProjects\AlzheimerML\Alzheimer\Entrenamiento\svm_model.pkl"
joblib.dump(svm_model, model_path)
print(f"Modelo guardado en: {model_path}")
