import cv2
import os
import numpy as np

# Ruta al dataset (corrige si las carpetas están directamente en "Entrenamiento")
dataset_path = r"C:\Users\Technologic PC\PycharmProjects\AlzheimerML\Alzheimer\Entrenamiento\train"

# Verificar existencia de la ruta principal
if not os.path.exists(dataset_path):
    print(f"La ruta al dataset no existe: {dataset_path}")
    exit()

# Función para cargar imágenes
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
print(f"Ejemplo de etiquetas: {labels[:5]}")
