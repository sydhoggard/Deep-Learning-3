"""
Script 2: Train SVHN Multi-Digit Model
- Loads preprocessed .npy files from Script 1
- Builds a CNN backbone with k+1 softmax heads (length + 5 digit heads)
- Trains with early stopping and learning-rate reduction
- Saves the best model to svhn_model.keras

Usage:
    python 2_train.py

Requirements:
    train_images.npy, train_labels.npy  (from 1_preprocess.py)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

# ── Config ──────────────────────────────────────────────────────────────────
IMG_SIZE   = 96          # must match 1_preprocess.py
MAX_DIGITS = 5           # number of digit heads
NUM_CLASSES_DIGIT  = 11  # digits 1-9, 10 (=0), plus null class 0
NUM_CLASSES_LENGTH = 6   # 0-5 digits (0 means "no number" / padding)
BATCH_SIZE = 64
EPOCHS     = 10
LR         = 1e-3
MODEL_PATH = "svhn_model.keras"
# ────────────────────────────────────────────────────────────────────────────


# ── 1. Load data ─────────────────────────────────────────────────────────────
print("Loading data ...")
X_train = np.load("train_images.npy").astype("float32") / 255.0
y_train = np.load("train_labels.npy")          # shape (N, MAX_DIGITS+1)

# Split off 10 % as validation
rng = np.random.default_rng(42)
idx = rng.permutation(len(X_train))
split = int(len(X_train) * 0.9)
train_idx, val_idx = idx[:split], idx[split:]

X_val   = X_train[val_idx];   X_train = X_train[train_idx]
y_val   = y_train[val_idx];   y_train = y_train[train_idx]

print(f"Train: {X_train.shape}  Val: {X_val.shape}")

# Separate label columns
def split_labels(y):
    """Returns a list: [length, d1, d2, d3, d4, d5]"""
    return [y[:, i] for i in range(MAX_DIGITS + 1)]

y_train_split = split_labels(y_train)
y_val_split   = split_labels(y_val)


# ── 2. Data augmentation ─────────────────────────────────────────────────────
augment = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomBrightness(0.15),
    layers.RandomContrast(0.15),
], name="augmentation")


# ── 3. Model architecture ─────────────────────────────────────────────────────
def build_model(img_size=IMG_SIZE):
    inputs = keras.Input(shape=(img_size, img_size, 3), name="image")
    x = augment(inputs)

    # ── Backbone: MobileNetV2 (pretrained on ImageNet, fine-tuned) ──
    base = keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    # Unfreeze top 40 layers for fine-tuning
    for layer in base.layers[:-40]:
        layer.trainable = False
    for layer in base.layers[-40:]:
        layer.trainable = True

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)

    # ── Output heads ──
    # Head 0: how many digits (0-5)
    length_out = layers.Dense(NUM_CLASSES_LENGTH, activation="softmax", name="length")(x)
    # Heads 1-5: each digit class (0=null, 1-9, 10)
    digit_outs = [
        layers.Dense(NUM_CLASSES_DIGIT, activation="softmax", name=f"digit_{i+1}")(x)
        for i in range(MAX_DIGITS)
    ]

    outputs = [length_out] + digit_outs
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


model = build_model()
model.summary()

# ── 4. Compile ────────────────────────────────────────────────────────────────
losses = {f"length": "sparse_categorical_crossentropy"}
losses.update({f"digit_{i+1}": "sparse_categorical_crossentropy" for i in range(MAX_DIGITS)})

# Weight digit losses more than length loss
loss_weights = {"length": 0.5}
loss_weights.update({f"digit_{i+1}": 1.0 for i in range(MAX_DIGITS)})

model.compile(
    optimizer=keras.optimizers.Adam(LR),
    loss=losses,
    loss_weights=loss_weights,
    metrics={k: "accuracy" for k in losses},
)

# ── 5. Callbacks ──────────────────────────────────────────────────────────────
callbacks = [
    keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        save_best_only=False,
        save_freq='epoch',
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=7, restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, verbose=1
    ),
]

# ── 6. Train ──────────────────────────────────────────────────────────────────
# Build y dict for Keras multi-output API
def make_y_dict(y_list):
    keys = ["length"] + [f"digit_{i+1}" for i in range(MAX_DIGITS)]
    return {k: v for k, v in zip(keys, y_list)}

history = model.fit(
    X_train, make_y_dict(y_train_split),
    validation_data=(X_val, make_y_dict(y_val_split)),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks,
)

print(f"\nBest model saved to {MODEL_PATH}")

# ── 7. Quick training curve summary ──────────────────────────────────────────
import json
with open("training_history.json", "w") as f:
    json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)
print("Training history saved to training_history.json")
