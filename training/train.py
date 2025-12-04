# training/train.py
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "app"))
from config import (
    TRAIN_DIR,
    VAL_DIR,
    MODEL_PATH,
    IMG_HEIGHT,
    IMG_WIDTH,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
)

def create_generators():
    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(f"TRAIN_DIR introuvable : {TRAIN_DIR}")
    if not os.path.isdir(VAL_DIR):
        raise FileNotFoundError(f"VAL_DIR introuvable : {VAL_DIR}")

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True,
    )

    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )

    print("Classes trouvées (train) :", train_gen.class_indices)
    return train_gen, val_gen

def build_model():
    base_model = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    )
    base_model.trainable = False  # on gèle le backbone au début

    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model

def main():
    train_gen, val_gen = create_generators()
    model = build_model()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
        ),
    ]

    model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
    )

    # Sauvegarde finale (au cas où)
    model.save(MODEL_PATH)
    print(f"✅ Modèle sauvegardé dans : {MODEL_PATH}")

if __name__ == "__main__":
    main()
