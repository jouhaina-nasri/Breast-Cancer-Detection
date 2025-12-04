import os

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

# Directory containing this file → .../BREAST-CANCER-DETECTION/app
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Project root → .../BREAST-CANCER-DETECTION
BASE_DIR = os.path.dirname(CURRENT_DIR)

# Data directories
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "training_set")
VAL_DIR   = os.path.join(DATA_DIR, "test_set")

# ---------------------------------------------------------
# MODEL PATH
# ---------------------------------------------------------

# Keras 3 model storage directory
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_NAME = "breast_cancer_cnn.keras"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# Ensure the model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------
# IMAGE CONFIGURATION
# ---------------------------------------------------------
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# ---------------------------------------------------------
# TRAINING CONFIGURATION
# ---------------------------------------------------------
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4

# ---------------------------------------------------------
# CLASS LABELS
# ---------------------------------------------------------
CLASS_LABELS = {
    0: "Cancer",   # malignant
    1: "Normal",
}

# ---------------------------------------------------------
# FILE VALIDATION
# ---------------------------------------------------------
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# ---------------------------------------------------------
# FLASK SECRET KEY
# ---------------------------------------------------------
SECRET_KEY = "change-me-in-production"
