# RSNA Pneumonia Detection — VS Code Project

## What This Project Does

- Loads a balanced sample of chest X-rays from the RSNA Pneumonia Detection Challenge.
- Trains a simple CNN and a MobileNetV2 transfer-learning model.
- Saves trained models, classification metrics, and confusion-matrix plots under `outputs/`.
- Supports pneumonia prediction on individual chest X-ray images.

**Important:** This is an educational prototype, not a clinically validated diagnostic system.

## 1. Install Software

1. Install Python 3.11 (recommended): https://www.python.org/downloads/
2. Install Visual Studio Code: https://code.visualstudio.com/
3. In VS Code, install the Microsoft Python extension.
4. Ensure you have enough disk space. The DICOM dataset is large.

## 2. Get the Dataset

Download the RSNA Pneumonia Detection Challenge dataset from Kaggle:

https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data

You may need a Kaggle account and must accept the competition rules.

Download:
- `stage_2_train_labels.csv.zip` (extract the CSV).
- `stage_2_train_images.zip` (extract the DICOM images; large file).

Arrange the files like this:

```text
RSNA_Pneumonia_VSCode_Project/
    train.py
    predict.py
    requirements.txt
    README.md
    data/
        stage_2_train_labels.csv
        stage_2_train_images/
            patient_id_1.dcm
            patient_id_2.dcm
            ...
```

The image folder must contain files named `<patientId>.dcm`.

**Do not upload the large dataset to GitHub.**

## 3. Open the Project

In VS Code, choose **File → Open Folder** and open `RSNA_Pneumonia_VSCode_Project`.

Open **Terminal → New Terminal** and run:

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Run Training

Start with a smaller sample to test the setup:

```powershell
python train.py --labels data/stage_2_train_labels.csv --images data/stage_2_train_images --per-class 100 --epochs 2
```

Then, if it works and your computer has enough RAM, disk space, and computing resources, run the larger experiment:

```powershell
python train.py --labels data/stage_2_train_labels.csv --images data/stage_2_train_images --per-class 1000 --epochs 5
```

## 4. Run Prediction on a Chest X-ray

After training the MobileNetV2 model, use `predict.py` to predict whether a chest X-ray contains a pneumonia finding.

**Note:** The trained model file `outputs/best_pneumonia_model.keras` must exist before running predictions.

### For DICOM Images

```powershell
python predict.py --image "data/stage_2_train_images/sample.dcm"
```

Replace `sample.dcm` with the actual DICOM filename.

### For JPG or PNG Images

```powershell
python predict.py --image "data/sample.png"
```

Replace the path with the actual image location.

The prediction script:
- Loads the trained MobileNetV2 model.
- Preprocesses the input chest X-ray.
- Calculates the pneumonia probability.
- Displays the prediction using a 50% threshold.

**Important:** This project is an educational prototype and is not intended for medical diagnosis.

## 5. Output Files

The training script creates an `outputs/` folder containing:

- `pneumonia_cnn_model.keras`
- `mobilenetv2_pneumonia_model.keras`
- `best_pneumonia_model.keras`
- `model_results.csv`
- CNN confusion-matrix PNG
- MobileNetV2 confusion-matrix PNG
- MobileNetV2 training-curve PNG

## 6. Troubleshooting

- **FileNotFoundError:** Verify the CSV and image-folder paths.
- **TensorFlow installation error:** Use Python 3.11 64-bit and a compatible package version.
- **Slow training:** Start with `--per-class 100 --epochs 2`. Training is slower without GPU acceleration.
- **Dataset too large:** Extract the image ZIP to a drive with sufficient free space.
- **Missing Kaggle data:** Sign in to Kaggle, accept the competition rules, and download the official dataset.

## 7. GitHub

This project is maintained in a private GitHub repository for academic submission and collaboration.

The repository should contain:
- Source code
- README and setup instructions
- Required project documentation
- Relevant model outputs needed to reproduce the prediction demo

Do not upload the original dataset or virtual environment. Share repository access with the instructor and assigned teaching assistants.
