
from pathlib import Path
import argparse
import random

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pydicom
import tensorflow as tf

from PIL import Image
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
from tensorflow.keras import layers, models


SEED = 42
IMAGE_SIZE = 224

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_balanced_sample(labels_csv, image_dir, per_class=1000):
    """Load a balanced set of DICOM images."""

    labels = pd.read_csv(labels_csv)

    required = {"patientId", "Target"}

    if not required.issubset(labels.columns):
        raise ValueError(f"CSV must contain columns: {required}")

    patient_labels = labels.groupby(
        "patientId", as_index=False
    )["Target"].max()

    positive = patient_labels[patient_labels["Target"] == 1]
    negative = patient_labels[patient_labels["Target"] == 0]

    if len(positive) < per_class or len(negative) < per_class:
        raise ValueError(
            f"Not enough examples for per_class={per_class}. "
            f"Available positive={len(positive)}, "
            f"negative={len(negative)}."
        )

    sample = pd.concat([
        positive.sample(n=per_class, random_state=SEED),
        negative.sample(n=per_class, random_state=SEED),
    ]).sample(
        frac=1,
        random_state=SEED
    ).reset_index(drop=True)

    images = []
    targets = []
    kept_ids = []

    image_dir = Path(image_dir)

    for idx, row in sample.iterrows():

        patient_id = str(row["patientId"])
        image_path = image_dir / f"{patient_id}.dcm"

        if not image_path.exists():
            print(f"Skipping missing image: {image_path}")
            continue

        try:
            dicom = pydicom.dcmread(str(image_path))

            arr = dicom.pixel_array.astype(np.float32)

            lo = float(arr.min())
            hi = float(arr.max())

            if hi > lo:
                arr = (arr - lo) / (hi - lo)
            else:
                arr = np.zeros_like(arr, dtype=np.float32)

            arr = np.asarray(
                Image.fromarray(arr).resize(
                    (IMAGE_SIZE, IMAGE_SIZE),
                    resample=Image.Resampling.BILINEAR,
                ),
                dtype=np.float32,
            )

            # Keep pixel values within the normalized range.
            arr = np.clip(arr, 0.0, 1.0)

            images.append(arr[..., np.newaxis])
            targets.append(int(row["Target"]))
            kept_ids.append(patient_id)

        except Exception as exc:
            print(
                f"Skipping unreadable DICOM {patient_id}: {exc}"
            )

        if (idx + 1) % 100 == 0:
            print(
                f"Checked {idx + 1}/{len(sample)} selected records"
            )

    if not images:
        raise RuntimeError(
            "No images loaded. Check the DICOM folder and CSV."
        )

    X = np.asarray(images, dtype=np.float32)
    y = np.asarray(targets, dtype=np.int32)
    ids = np.asarray(kept_ids)

    print(f"Loaded images: {len(X)}")
    print(f"Tensor shape: {X.shape}")

    unique, counts = np.unique(y, return_counts=True)

    print("Class counts:", dict(zip(unique, counts)))

    return X, y, ids


def build_cnn():
    """Build the baseline CNN model."""

    model = models.Sequential([
        layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
        layers.Conv2D(16, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    return model


def build_mobilenet():
    """Build MobileNetV2 transfer-learning model."""

    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )

    base.trainable = False

    inputs = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    return model


def evaluate_and_save(model, X_test, y_test, out_dir, name):
    """Evaluate model and save its confusion matrix."""

    probs = model.predict(X_test, verbose=0).ravel()
    preds = (probs >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, preds)

    print(f"\n{name} test accuracy: {accuracy:.4f}")

    print(
        classification_report(
            y_test,
            preds,
            target_names=[
                "No pneumonia finding",
                "Pneumonia finding",
            ],
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_test, preds)

    print("Confusion matrix:\n", cm)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No pneumonia", "Pneumonia"],
    )

    disp.plot(values_format="d")

    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()

    output_file = (
        out_dir / f"{name.lower()}_confusion_matrix_1000.png"
    )

    plt.savefig(str(output_file), dpi=160, format="png")
    plt.close()

    print(f"Confusion matrix saved: {output_file.resolve()}")

    return {
        "model": name,
        "accuracy": accuracy,
        "pneumonia_precision": precision_score(
            y_test, preds, zero_division=0
        ),
        "pneumonia_recall": recall_score(
            y_test, preds, zero_division=0
        ),
        "pneumonia_f1": f1_score(
            y_test, preds, zero_division=0
        ),
    }


def save_training_curves(histories, out_dir):
    """Save accuracy and loss graphs for both models."""

    for model_name, history in histories.items():

        name = model_name.lower()

        plt.figure(figsize=(8, 5))

        plt.plot(
            history.history["accuracy"],
            label="Training accuracy",
        )

        plt.plot(
            history.history["val_accuracy"],
            label="Validation accuracy",
        )

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{model_name} Training vs Validation Accuracy")
        plt.legend()
        plt.tight_layout()

        accuracy_file = out_dir / f"{name}_accuracy_curve_1000.png"

        plt.savefig(str(accuracy_file), dpi=160, format="png")
        plt.close()

        print(f"Accuracy graph saved: {accuracy_file.resolve()}")

        plt.figure(figsize=(8, 5))

        plt.plot(
            history.history["loss"],
            label="Training loss",
        )

        plt.plot(
            history.history["val_loss"],
            label="Validation loss",
        )

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{model_name} Training vs Validation Loss")
        plt.legend()
        plt.tight_layout()

        loss_file = out_dir / f"{name}_loss_curve_1000.png"

        plt.savefig(str(loss_file), dpi=160, format="png")
        plt.close()

        print(f"Loss graph saved: {loss_file.resolve()}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--labels",
        default="data/stage_2_train_labels.csv",
        help="Path to stage_2_train_labels.csv",
    )

    parser.add_argument(
        "--images",
        default="data/stage_2_train_images",
        help="Folder containing DICOM images",
    )

    parser.add_argument(
        "--per-class",
        type=int,
        default=1000,
        help="Number of positive and negative examples",
    )

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)

    args = parser.parse_args()

    # Use a separate folder to preserve previous results.
    out_dir = Path("outputs_corrected")
    out_dir.mkdir(exist_ok=True)

    labels_path = Path(args.labels)
    images_path = Path(args.images)

    if not labels_path.exists():
        raise FileNotFoundError(
            f"Labels CSV not found: {labels_path.resolve()}"
        )

    if not images_path.is_dir():
        raise FileNotFoundError(
            f"Image folder not found: {images_path.resolve()}"
        )

    print("\n========== LOADING DATASET ==========")

    X, y, patient_ids = load_balanced_sample(
        labels_path,
        images_path,
        per_class=args.per_class,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )

    print("\nTrain:", X_train.shape)
    print("Test:", X_test.shape)

    print("\n========== CNN TRAINING ==========")

    cnn = build_cnn()

    cnn_history = cnn.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=2,
                restore_best_weights=True,
            )
        ],
        verbose=1,
    )

    cnn.save(out_dir / "pneumonia_cnn_model.keras")

    cnn_result = evaluate_and_save(
        cnn, X_test, y_test, out_dir, "CNN"
    )

    X_train_rgb = np.repeat(X_train, 3, axis=-1) * 2.0 - 1.0
    X_test_rgb = np.repeat(X_test, 3, axis=-1) * 2.0 - 1.0

    print("\n========== MOBILENETV2 TRAINING ==========")

    mobile = build_mobilenet()

    mobile_history = mobile.fit(
        X_train_rgb,
        y_train,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=2,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                out_dir / "best_pneumonia_model.keras",
                monitor="val_loss",
                save_best_only=True,
            ),
        ],
        verbose=1,
    )

    mobile.save(out_dir / "mobilenetv2_pneumonia_model.keras")

    mobile_result = evaluate_and_save(
    mobile, X_test_rgb, y_test, out_dir, "MobileNetV2"
    )

    results_file = out_dir / "model_results.csv"

    pd.DataFrame([
        cnn_result,
        mobile_result,
    ]).to_csv(results_file, index=False)

    print(f"\nModel results saved: {results_file.resolve()}")

    histories = {
        "CNN": cnn_history,
        "MobileNetV2": mobile_history,
    }

    print("\n========== SAVING TRAINING GRAPHS ==========")

    save_training_curves(histories, out_dir)

    print("\n========== PROJECT TRAINING COMPLETED ==========")
    print("Check outputs_corrected for models, metrics and graphs.")
    print(
        "Reminder: educational prototype only; "
        "not for medical diagnosis."
    )


if __name__ == "__main__":
    main()

