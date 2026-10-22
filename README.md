# RSNA Pneumonia Detection — VS Code Project

## What this project does
- Loads a balanced sample of chest X-rays from the RSNA Pneumonia Detection Challenge.
- Trains a simple CNN and a MobileNetV2 transfer-learning model.
- Saves trained models, classification metrics, and confusion-matrix plots under `outputs/`.

**Important:** This is an educational prototype, not a clinically validated diagnostic system.

## 1. Install software
1. Install Python 3.11 (recommended) from https://www.python.org/downloads/
2. Install Visual Studio Code from https://code.visualstudio.com/
3. In VS Code, install the Microsoft **Python** extension.
4. Ensure you have enough disk space. The DICOM dataset is large.

## 2. Get the dataset
You need the RSNA Pneumonia Detection Challenge dataset from Kaggle:
https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data

You may need a Kaggle account and to accept the competition rules. Download:
- `stage_2_train_labels.csv.zip` (extract the CSV)
- `stage_2_train_images.zip` (extract the DICOM images; large file)

Arrange the files like this:
```
RSNA_Pneumonia_VSCode_Project/
  train.py
  requirements.txt
  README.md
  data/
    stage_2_train_labels.csv
    stage_2_train_images/
      000a... .dcm
      ...
```
The image folder must contain files named `<patientId>.dcm`. Do not upload the large dataset to GitHub.

## 3. Open the project
In VS Code choose **File → Open Folder** and open `RSNA_Pneumonia_VSCode_Project`.

Open ** → New Terminal** and run the commands below.

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

### Run training
Start with a smaller sample to test the setup:
```powershell
python train.py --labels data/stage_2_train_labels.csv --images data/stage_2_train_images --per-class 100 --epochs 2
```
Then, if it works and your computer has enough RAM/disk/compute, run the 2,000-image experiment:
```powershell
python train.py --labels data/stage_2_train_labels.csv --images data/stage_2_train_images --per-class 1000 --epochs 5
```

## 4. Output files
The script creates `outputs/` containing:
- `pneumonia_cnn_model.keras`
- `mobilenetv2_pneumonia_model.keras`
- `best_pneumonia_model.keras`
- `model_results.csv`
- confusion-matrix PNGs
- MobileNetV2 training-curve PNG

## Troubleshooting
- **FileNotFoundError:** verify the CSV and image folder paths.
- **TensorFlow install error:** use Python 3.11 64-bit and a supported version of pip.
- **Slow training:** use `--per-class 100 --epochs 2` first; training is much slower without a supported GPU.
- **Dataset too large:** extract the image ZIP to a drive with sufficient free space.
- **Missing Kaggle data:** sign in to Kaggle, accept the competition rules, and download from the official competition data page.

## GitHub
GitHub upload is optional unless your instructor asks for it. If you do upload, exclude the dataset, virtual environment, and generated model files. Add a `.gitignore` before committing.
