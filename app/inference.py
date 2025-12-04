import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import (
    IMG_HEIGHT,
    IMG_WIDTH,
    CLASS_LABELS,
    ALLOWED_EXTENSIONS,
    VAL_DIR,
)

# ---------------------------------------------------------
# FILE VALIDATION
# ---------------------------------------------------------
def allowed_file(filename: str) -> bool:
    """Check if the uploaded file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------
# IMAGE PREPROCESSING
# ---------------------------------------------------------
def preprocess_image_file(file_storage):
    """
    Reads and preprocesses an uploaded image for MobileNetV2 inference.
    """
    file_bytes = np.frombuffer(file_storage.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Unable to read the image file.")

    # Convert BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to expected input size
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

    img = img.astype("float32")
    img = preprocess_input(img)  # MobileNetV2 preprocessing

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img


# ---------------------------------------------------------
# SINGLE IMAGE PREDICTION
# ---------------------------------------------------------
def predict_image(model, file_storage):
    """
    Runs inference on a single image:
      - model output is a sigmoid scalar in [0,1]
      - value represents P(y=1 = Normal)
    """
    img = preprocess_image_file(file_storage)
    preds = model.predict(img, verbose=0)[0][0]  # scalar sigmoid output

    prob_normal = float(preds)
    class_idx = 1 if prob_normal >= 0.5 else 0

    label = CLASS_LABELS[class_idx]

    return label, prob_normal


# ---------------------------------------------------------
# FULL TEST SET EVALUATION
# ---------------------------------------------------------
def evaluate_model_on_test_set(model):
    """
    Evaluates the model on data/test_set and returns:
        - accuracy
        - confusion matrix components (TP, TN, FP, FN)
        - number of samples
    """
    if not VAL_DIR or not os.path.exists(VAL_DIR):
        raise ValueError("VAL_DIR is not set correctly in config.py.")

    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    test_gen = datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=32,
        class_mode="binary",
        shuffle=False,
    )

    # Ground truth labels: 0 = Cancer, 1 = Normal
    y_true = test_gen.classes

    # Predictions from model
    preds = model.predict(test_gen, verbose=0).ravel()
    y_pred = (preds >= 0.5).astype(int)

    # Confusion matrix
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    accuracy = float((y_true == y_pred).mean())

    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": accuracy,
        "n_samples": int(len(y_true)),
    }
