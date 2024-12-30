# Alzheimer MRI Analysis and Classification

This repository contains a Machine Learning pipeline for analyzing MRI images and classifying them into different stages of Alzheimer’s Disease:

- **No Impairment**
- **Very Mild Impairment**
- **Mild Impairment**
- **Moderate Impairment**

The project utilizes the dataset provided by [Luke Chugh on Kaggle](https://www.kaggle.com/datasets/lukechugh/best-alzheimer-mri-dataset-99-accuracy?resource=download). Full credit for the dataset goes to its author.

## Project Workflow

### 1. **Dataset Preparation**

The dataset is structured into training and testing directories, with subfolders for each classification category. Preprocessing includes:

- Converting images to grayscale.
- Resizing images to a consistent resolution (256x256).

### 2. **Feature Extraction**

Statistical features were extracted from each image, such as:

- **Number of isolated groups**
- **Average area of groups**
- **Maximum and minimum group area**
- **Standard deviation of group areas**

These features were used as inputs for the classification model.

### 3. **Model Training**

A **Support Vector Machine (SVM)** model was trained to classify the images. The training process included:

- Normalizing the extracted features.
- Encoding labels numerically.
- Evaluating the model using precision, recall, and F1-score.

### 4. **Model Evaluation and Prediction**

The trained model can classify new, unseen MRI images into one of the Alzheimer’s stages by analyzing the extracted features.

## Libraries Used

The following Python libraries were used in this project:

- **OpenCV:** Image preprocessing and contour analysis.
- **NumPy:** Numerical computations.
- **pandas:** Handling extracted features and exporting results.
- **matplotlib:** Visualizing preprocessed images and results.
- **scikit-learn:**
  - Normalization (`StandardScaler`)
  - Label encoding (`LabelEncoder`)
  - Model training (`SVC` for SVM)
- **joblib:** Saving and loading trained models, scalers, and encoders.

## Installation

To run the project locally, install the required dependencies:

```bash
pip install opencv-python opencv-python-headless numpy pandas matplotlib scikit-learn joblib
```

## Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/AlzheimerML.git
   ```
2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/lukechugh/best-alzheimer-mri-dataset-99-accuracy?resource=download) and place it in the project directory.
3. Run the scripts in the following order:
   - `1CargaImagen.py`: Prepares and preprocesses the images.
   - `5DatasetPrep.py`: Extracts features and prepares the dataset.
   - `6ModeloSVM.py`: Trains the SVM model.

## Results

The model achieved an accuracy of approximately 38% with the current features and preprocessing steps. Future improvements may include:

- Optimizing hyperparameters of the SVM.
- Experimenting with other models (e.g., Random Forest, Neural Networks).
- Enhancing feature extraction techniques.

## Credits

This project uses the **Best Alzheimer MRI Dataset** by [Luke Chugh](https://www.kaggle.com/datasets/lukechugh/best-alzheimer-mri-dataset-99-accuracy?resource=download). Full credit goes to the dataset author for making it publicly available.

## License

This project is open-source and available under the [MIT License](LICENSE). Please ensure proper attribution if you use this code or pipeline in your own work.

