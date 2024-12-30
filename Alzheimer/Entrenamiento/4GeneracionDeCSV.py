import cv2
import os
import numpy as np
import pandas as pd

# Ruta al dataset
dataset_path = r"C:\Users\Technologic PC\PycharmProjects\AlzheimerML\Alzheimer\Entrenamiento\train"

# Verificar existencia de la ruta principal
if not os.path.exists(dataset_path):
    print(f"La ruta al dataset no existe: {dataset_path}")
    exit()

# Función para cargar y preprocesar imágenes
def load_and_preprocess_images(path):
    images = []
    labels = []
    categories = ["No Impairment", "Very Mild Impairment", "Mild Impairment", "Moderate Impairment"]

    for category in categories:
        category_path = os.path.join(path, category)
        if not os.path.exists(category_path):
            print(f"La carpeta no existe: {category_path}")
            continue

        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (256, 256))
                images.append(img)
                labels.append(category)
            except Exception as e:
                print(f"Error al cargar la imagen: {img_path}, Error: {e}")
    return np.array(images), labels

# Cargar imágenes
images, labels = load_and_preprocess_images(dataset_path)
print(f"Total de imágenes cargadas: {len(images)}")
if len(labels) > 0:
    print(f"Ejemplo de etiquetas: {labels[:5]}")
else:
    print("No se cargaron etiquetas. Revisa la estructura del dataset.")

# Si no hay imágenes cargadas, finaliza la ejecución
if len(images) == 0:
    exit()

# Análisis de imágenes
def analyze_image(image):
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_groups = len(contours)
    areas = [cv2.contourArea(cnt) for cnt in contours]
    if areas:
        return {
            "num_groups": num_groups,
            "area_avg": np.mean(areas),
            "area_max": np.max(areas),
            "area_min": np.min(areas),
            "area_std": np.std(areas)
        }
    return {"num_groups": 0, "area_avg": 0, "area_max": 0, "area_min": 0, "area_std": 0}

# Calcular estadísticas
stats = []
for img, label in zip(images, labels):
    stats_data = analyze_image(img)
    stats_data["label"] = label
    stats.append(stats_data)

# Convertir a DataFrame y guardar en CSV
stats_df = pd.DataFrame(stats)
output_path = r"C:\Users\Technologic PC\PycharmProjects\AlzheimerML\Alzheimer\Entrenamiento\image_stats.csv"
stats_df.to_csv(output_path, index=False)
print(f"Archivo CSV guardado en: {output_path}")
