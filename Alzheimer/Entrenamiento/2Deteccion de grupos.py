import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Ruta al dataset
dataset_path = r"C:\Users\Technologic PC\PycharmProjects\AlzheimerML\Alzheimer\Entrenamiento\train"


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
            # Leer imagen
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

# Imprimir información de depuración
print(f"Total de imágenes cargadas: {len(images)}")
print(f"Ejemplo de etiquetas: {labels[:5]}")

def segment_and_analyze(image):
    # Aplicar un umbral binario
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Detectar contornos
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Contar las áreas aisladas
    num_groups = len(contours)

    # Calcular áreas de cada contorno
    areas = [cv2.contourArea(cnt) for cnt in contours]

    # Retornar resultados
    return num_groups, areas


# Analizar las imágenes cargadas
results = []
for img in images:
    num_groups, areas = segment_and_analyze(img)
    results.append({"num_groups": num_groups, "areas": areas})

# Ejemplo de salida
print(f"Resultados de la primera imagen: {results[0]}")

import random


def segment_and_visualize(image):
    # Aplicar un umbral binario
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Detectar contornos
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Crear una copia de la imagen original en color
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Dibujar cada contorno con un color aleatorio
    for cnt in contours:
        color = [random.randint(0, 255) for _ in range(3)]  # Generar un color RGB aleatorio
        cv2.drawContours(color_image, [cnt], -1, color, 2)

    # Contar las áreas aisladas
    num_groups = len(contours)

    return color_image, num_groups


# Procesar y visualizar una imagen como ejemplo
index = 0  # Cambia el índice para probar con otras imágenes
colored_image, num_groups = segment_and_visualize(images[index])

print(f"Número de grupos detectados: {num_groups}")

# Mostrar la imagen con los grupos coloreados
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(colored_image, cv2.COLOR_BGR2RGB))  # Convertir BGR a RGB para Matplotlib
plt.title(f"Grupos detectados: {num_groups}")
plt.axis('off')
plt.show()