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
