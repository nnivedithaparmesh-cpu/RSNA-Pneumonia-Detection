# RSNA Pneumonia Detection Using Deep Learning

## 1. Project Overview

The **RSNA Pneumonia Detection** project uses deep learning techniques to classify chest X-ray images and identify possible pneumonia findings.

The project uses the RSNA Pneumonia Detection Challenge dataset and implements two deep learning approaches:

- Custom Convolutional Neural Network (CNN)
- MobileNetV2 Transfer Learning

The models are trained and evaluated using chest X-ray images. The project also provides a prediction script that accepts individual DICOM, JPG, and PNG images.

**Important:** This project is an educational prototype and is not clinically validated. It is not intended to replace professional medical diagnosis.

## 2. Objectives

- Develop a deep learning-based pneumonia classification system.
- Preprocess chest X-ray images for model training and prediction.
- Implement and compare CNN and MobileNetV2 architectures.
- Evaluate model performance using accuracy, precision, recall, F1-score, and confusion matrices.
- Perform predictions on individual chest X-ray images.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| TensorFlow / Keras | Deep learning framework |
| MobileNetV2 | Transfer learning model |
| CNN | Image classification |
| NumPy | Numerical operations |
| Pandas | Dataset handling |
| Pydicom | Reading DICOM images |
| Matplotlib | Graphs and visualizations |
| Pillow (PIL) | Image processing |
| Scikit-learn | Model evaluation |
| VS Code | Development environment |

## 4. Dataset

The project uses the **RSNA Pneumonia Detection Challenge** dataset.

Dataset source:

https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data

The dataset contains chest X-ray images in DICOM format and corresponding labels.

The labels indicate whether a pneumonia finding is present.

Download the dataset from Kaggle after signing in and accepting the competition rules.

### Expected Dataset Structure

```text
data/
    stage_2_train_labels.csv
    stage_2_train_images/
        patient_id_1.dcm
        patient_id_2.dcm
        ...
```

The dataset is not included in this repository because of its large size.

## 5. Project Structure

```text
RSNA_Pneumonia_VSCode_Project/
│
├── data/
│   ├── stage_2_train_labels.csv
│   └── stage_2_train_images/
│
├── outputs/
│   ├── best_pneumonia_model.keras
│   ├── best_pneumonia_model_compatible.keras
│   ├── pneumonia_cnn_model.keras
│   ├── mobilenetv2_pneumonia_model.keras
│   ├── model_results.csv
│   ├── cnn_confusion_matrix.png
│   ├── mobilenetv2_confusion_matrix.png
│   └── mobilenetv2_training_curves.png
│
├── outputs_small_test/
│
├── train.py
├── predict.py
├── fix_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

The `.venv` virtual environment and Python cache files are not required in the GitHub repository.

## 6. Model Development

### A. Custom CNN

A Convolutional Neural Network is implemented for pneumonia classification.

The CNN learns image features through convolution and pooling layers and performs binary classification using a final output layer.

### B. MobileNetV2

MobileNetV2 is used as a transfer learning model.

The pretrained ImageNet feature extractor is used with a custom classification head for pneumonia detection.

The model uses:

- Pretrained MobileNetV2 architecture
- Frozen base feature extractor
- Global average pooling
- Dropout
- Dense output layer with sigmoid activation

MobileNetV2 achieved better overall performance than the custom CNN in the conducted experiment.

## 7. Image Preprocessing

The input chest X-ray images undergo preprocessing before being passed to the model.

The preprocessing steps include:

1. Reading the input DICOM or image file.
2. Extracting the image pixel data.
3. Converting the image to grayscale.
4. Normalizing pixel values.
5. Resizing the image to 224 × 224 pixels.
6. Converting grayscale images into three-channel images.
7. Scaling pixel values to the range [-1, 1].

The processed image is then passed to the trained MobileNetV2 model.

## 8. Model Evaluation

The models are evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

### Experimental Results

The following results were obtained from the balanced 400-image test set.

| Evaluation Metric | CNN | MobileNetV2 |
|---|---:|---:|
| Accuracy | 54.00% | 73.25% |
| Pneumonia Precision | 52.56% | 72.25% |
| Pneumonia Recall | 82.00% | 75.50% |
| Pneumonia F1-score | 64.06% | 73.84% |

### Result Analysis

- The custom CNN achieved 54% accuracy.
- MobileNetV2 achieved 73.25% accuracy.
- MobileNetV2 improved accuracy by 19.25 percentage points.
- MobileNetV2 achieved a higher F1-score and precision.
- The custom CNN achieved higher pneumonia recall.

Based on these experimental results, MobileNetV2 performed better overall on the selected test set.

**Note:** These results are from a limited balanced test sample and should not be interpreted as clinical performance or generalization to real-world patient populations.

## 9. Installation and Setup

### Prerequisites

- Python 3.11 (recommended by the original project setup)
- Visual Studio Code
- Required Python packages
- RSNA dataset

### Step 1: Clone the Repository

```bash
git clone https://github.com/nnivedithaparmesh-cpu/RSNA-Pneumonia-Detection.git
```

### Step 2: Open the Project Folder

```bash
cd RSNA-Pneumonia-Detection
```

### Step 3: Create a Virtual Environment

```powershell
py -3.11 -m venv .venv
```

### Step 4: Activate the Virtual Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again.

### Step 5: Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 10. Training the Models

Ensure that the dataset is arranged in the expected directory structure.

Run the training script using:

```powershell
python train.py --labels data/stage_2_train_labels.csv --images data/stage_2_train_images --per-class 100 --epochs 2
```

For a larger experiment, use:

```powershell
python train.py --labels data/stage_2_train_labels.csv --images data/stage_2_train_images --per-class 1000 --epochs 5
```

Training time depends on available system resources.

The training process saves the trained models, evaluation metrics, and visualization files in the `outputs/` directory.

**Note:** The larger training command is optional. Existing trained models can be used directly for prediction without retraining.

## 11. Model Compatibility

The saved model may contain older Keras configuration fields that are not supported by the currently installed Keras version.

The `fix_model.py` script creates a compatible copy of the original trained model.

Run:

```powershell
python fix_model.py
```

The script creates:

```text
outputs/best_pneumonia_model_compatible.keras
```

The original trained model remains unchanged.

## 12. Running Predictions

The `predict.py` script performs pneumonia classification on an individual chest X-ray image.

The prediction script uses the compatible MobileNetV2 model:

```text
outputs/best_pneumonia_model_compatible.keras
```

### A. Prediction Using a DICOM Image

```powershell
python predict.py --image "data/stage_2_train_images/sample.dcm"
```

### B. Prediction Using a JPG or PNG Image

```powershell
python predict.py --image "data/sample.png"
```

Replace the sample filename with the actual image path.

### Prediction Output

The program displays:

- Pneumonia probability
- Classification result
- Educational-use disclaimer

A probability threshold of 0.5 (50%) is used:

- Probability greater than or equal to 50%: Pneumonia finding detected.
- Probability below 50%: No pneumonia finding detected.

These are model-generated classifications, not medical diagnoses.

## 13. Output Files

The `outputs/` directory contains:

| File | Description |
|---|---|
| `pneumonia_cnn_model.keras` | Trained CNN model |
| `mobilenetv2_pneumonia_model.keras` | Trained MobileNetV2 model |
| `best_pneumonia_model.keras` | Original best model |
| `best_pneumonia_model_compatible.keras` | Compatible model copy for prediction |
| `model_results.csv` | Model evaluation metrics |
| `cnn_confusion_matrix.png` | CNN confusion matrix |
| `mobilenetv2_confusion_matrix.png` | MobileNetV2 confusion matrix |
| `mobilenetv2_training_curves.png` | Training and validation accuracy graph |

## 14. Limitations

- The project uses a limited balanced dataset sample for experimentation.
- Model performance may vary with different datasets and image quality.
- The model is not clinically validated.
- False positive and false negative predictions are possible.
- The model should not be used for actual medical diagnosis or treatment decisions.

## 15. Future Enhancements

- Train and evaluate on larger and more diverse datasets.
- Improve model generalization.
- Explore advanced CNN architectures.
- Develop a user-friendly web interface.
- Add explainable AI techniques such as Grad-CAM.
- Improve prediction performance and evaluation.

## 16. Conclusion

This project demonstrates the application of deep learning and transfer learning for pneumonia classification using chest X-ray images.

Two approaches, a custom CNN and MobileNetV2, were implemented and compared.

MobileNetV2 achieved 73.25% test accuracy, outperforming the custom CNN in overall accuracy and F1-score on the selected test set.

The project provides an educational demonstration of image preprocessing, deep learning model development, model evaluation, and individual image prediction.

**This system is an educational prototype and is not intended for clinical use.**
