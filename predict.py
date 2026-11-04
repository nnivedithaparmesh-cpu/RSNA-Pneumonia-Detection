
import argparse
from pathlib import Path

import numpy as np
import pydicom
import tensorflow as tf
from PIL import Image

IMAGE_SIZE = 224


def load_xray(image_path):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if image_path.suffix.lower() == ".dcm":
        dicom = pydicom.dcmread(str(image_path))
        arr = dicom.pixel_array.astype(np.float32)
    else:
        arr = np.asarray(
            Image.open(image_path).convert("L"),
            dtype=np.float32
        )

    # Same normalization approach as the training script
    lo, hi = float(arr.min()), float(arr.max())

    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)

    # Resize to model input dimensions
    arr = np.asarray(
        Image.fromarray(arr).resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),
        dtype=np.float32
    )

    # Convert grayscale to RGB and scale to [-1, 1]
    arr = np.repeat(arr[..., np.newaxis], 3, axis=-1)
    arr = arr * 2.0 - 1.0

    return np.expand_dims(arr, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Predict pneumonia from a chest X-ray"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to a DICOM, JPG or PNG X-ray image"
    )

    args = parser.parse_args()

    model_path = Path("outputs/best_pneumonia_model.keras")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}"
        )

    print("Loading trained model...")
    model = tf.keras.models.load_model(model_path)

    print("Processing X-ray...")
    image = load_xray(args.image)

    probability = float(
        model.predict(image, verbose=0)[0][0]
    )

    print("\n========== PREDICTION ==========")
    print(f"Pneumonia probability: {probability:.2%}")

    if probability >= 0.5:
        print("Result: Pneumonia finding detected")
    else:
        print("Result: No pneumonia finding detected")

    print("================================")
    print("Educational prototype only.")
    print("Not intended for medical diagnosis.")


if __name__ == "__main__":
    main()
