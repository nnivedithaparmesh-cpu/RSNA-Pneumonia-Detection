
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
)

from train import load_balanced_sample, SEED, IMAGE_SIZE


def evaluate_model(model, X_test, y_test, name, output_dir):

    print(f"\n========== {name} EVALUATION ==========")

    probabilities = model.predict(
        X_test,
        batch_size=32,
        verbose=1,
    ).ravel()

    predictions = (probabilities >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, predictions)

    print(f"\nAccuracy: {accuracy:.4f}")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "No pneumonia finding",
                "Pneumonia finding",
            ],
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_test, predictions)

    print("Confusion matrix:\n", cm)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No pneumonia", "Pneumonia"],
    )

    display.plot(values_format="d")
    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()

    image_path = output_dir / f"{name.lower()}_confusion_matrix_1000.png"

    plt.savefig(image_path, dpi=160)
    plt.close()

    print(f"Saved: {image_path}")

    return {
        "model": name,
        "accuracy": accuracy,
        "pneumonia_precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "pneumonia_recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "pneumonia_f1": f1_score(
            y_test, predictions, zero_division=0
        ),
    }


def main():

    output_dir = Path("outputs_corrected")
    output_dir.mkdir(exist_ok=True)

    print("\n========== LOADING TEST DATA ==========")

    X, y, patient_ids = load_balanced_sample(
        "data/stage_2_train_labels.csv",
        "data/stage_2_train_images",
        per_class=1000,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )

    print("\nTest data shape:", X_test.shape)

    print("\n========== LOADING SAVED MODELS ==========")

    cnn = tf.keras.models.load_model(
        output_dir / "pneumonia_cnn_model.keras"
    )

    mobile = tf.keras.models.load_model(
        output_dir / "mobilenetv2_pneumonia_model.keras"
    )

    print("Both models loaded successfully.")

    print("\n========== CNN EVALUATION ==========")

    cnn_result = evaluate_model(
        cnn,
        X_test,
        y_test,
        "CNN",
        output_dir,
    )

    print("\n========== PREPARING MOBILENET INPUT ==========")

    X_test_rgb = (
        np.repeat(X_test, 3, axis=-1) * 2.0 - 1.0
    )

    print("MobileNetV2 test shape:", X_test_rgb.shape)

    mobile_result = evaluate_model(
        mobile,
        X_test_rgb,
        y_test,
        "MobileNetV2",
        output_dir,
    )

    results = pd.DataFrame([
        cnn_result,
        mobile_result,
    ])

    results_path = output_dir / "model_results.csv"

    results.to_csv(results_path, index=False)

    print("\n========== FINAL MODEL COMPARISON ==========")

    print(results.to_string(index=False))

    print(f"\nResults saved to: {results_path.resolve()}")

    print("\n========== EVALUATION COMPLETED ==========")

    print("No model training was performed.")
    print("Educational prototype only; not for medical diagnosis.")


if __name__ == "__main__":
    main()
