import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Ruta al dataset
dataset_path = r"C:\Users\Technologic PC\PycharmProjects\AlzheimerML\Alzheimer\1Preparacion\train"


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


def analyze_image(image):
    # Aplicar umbral binario
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Detectar contornos
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calcular métricas
    num_groups = len(contours)
    areas = [cv2.contourArea(cnt) for cnt in contours]

    if areas:  # Evitar errores si no hay áreas detectadas
        area_avg = np.mean(areas)
        area_max = np.max(areas)
        area_min = np.min(areas)
        area_std = np.std(areas)
    else:
        area_avg, area_max, area_min, area_std = 0, 0, 0, 0

    return {
        "num_groups": num_groups,
        "area_avg": area_avg,
        "area_max": area_max,
        "area_min": area_min,
        "area_std": area_std
    }


# Analizar todas las imágenes
stats = []
for img, label in zip(images, labels):
    image_stats = analyze_image(img)
    image_stats["label"] = label
    stats.append(image_stats)

# Mostrar resultados para las primeras imágenes
for i, stat in enumerate(stats[:5]):
    print(f"Imagen {i + 1}: {stat}")

import pandas as pd

# Convertir los resultados a un DataFrame
stats_df = pd.DataFrame(stats)

# Guardar el DataFrame como un archivo CSV
output_path = r"C:\Users\Technologic PC\PycharmProjects\AlzheimerML\Alzheimer\1Preparacion\4image_stats.csv"
stats_df.to_csv(output_path, index=False)

print(f"Archivo CSV guardado en: {output_path}")
