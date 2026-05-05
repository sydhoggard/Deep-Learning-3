import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.keras.backend.clear_session()

IMG_SIZE = 96
MAX_DIGITS = 5
NUM_CLASSES_LENGTH = 6
NUM_CLASSES_DIGIT = 11

BATCH_SIZE = 64
EPOCHS = 15
LR = 0.001
MODEL_PATH = "svhn_cnn_model.keras"

# Memory-map instead of loading everything fully into RAM
X = np.load("train_images.npy", mmap_mode="r")
y = np.load("train_labels.npy", mmap_mode="r")

N = len(X)
rng = np.random.default_rng(42)
idx = rng.permutation(N)

split = int(N * 0.9)
train_idx = idx[:split]
val_idx = idx[split:]

print("Train samples:", len(train_idx))
print("Val samples:", len(val_idx))

def make_dataset(indices, shuffle=False):
    def gen():
        for i in indices:
            image = X[i].astype("float32") / 255.0
            label = y[i]
            yield image, {
                "length": label[0],
                "digit_1": label[1],
                "digit_2": label[2],
                "digit_3": label[3],
                "digit_4": label[4],
                "digit_5": label[5],
            }

    output_signature = (
        tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
        {
            "length": tf.TensorSpec(shape=(), dtype=tf.int32),
            "digit_1": tf.TensorSpec(shape=(), dtype=tf.int32),
            "digit_2": tf.TensorSpec(shape=(), dtype=tf.int32),
            "digit_3": tf.TensorSpec(shape=(), dtype=tf.int32),
            "digit_4": tf.TensorSpec(shape=(), dtype=tf.int32),
            "digit_5": tf.TensorSpec(shape=(), dtype=tf.int32),
        }
    )

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    if shuffle:
        ds = ds.shuffle(2000)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(train_idx, shuffle=True)
val_ds = make_dataset(val_idx, shuffle=False)

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")

x = layers.RandomRotation(0.03)(inputs)
x = layers.RandomZoom(0.08)(x)
x = layers.RandomContrast(0.10)(x)

x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D(2)(x)

x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D(2)(x)

x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D(2)(x)

x = layers.Flatten()(x)

x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.4)(x)

x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)

length_out = layers.Dense(NUM_CLASSES_LENGTH, activation="softmax", name="length")(x)
digit_1 = layers.Dense(NUM_CLASSES_DIGIT, activation="softmax", name="digit_1")(x)
digit_2 = layers.Dense(NUM_CLASSES_DIGIT, activation="softmax", name="digit_2")(x)
digit_3 = layers.Dense(NUM_CLASSES_DIGIT, activation="softmax", name="digit_3")(x)
digit_4 = layers.Dense(NUM_CLASSES_DIGIT, activation="softmax", name="digit_4")(x)
digit_5 = layers.Dense(NUM_CLASSES_DIGIT, activation="softmax", name="digit_5")(x)

model = keras.Model(
    inputs=inputs,
    outputs=[length_out, digit_1, digit_2, digit_3, digit_4, digit_5]
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR),
    loss={
        "length": "sparse_categorical_crossentropy",
        "digit_1": "sparse_categorical_crossentropy",
        "digit_2": "sparse_categorical_crossentropy",
        "digit_3": "sparse_categorical_crossentropy",
        "digit_4": "sparse_categorical_crossentropy",
        "digit_5": "sparse_categorical_crossentropy",
    },
    loss_weights={
        "length": 0.5,
        "digit_1": 1.5,
        "digit_2": 1.5,
        "digit_3": 1.2,
        "digit_4": 1.0,
        "digit_5": 1.0,
    },
    metrics={
        "length": "accuracy",
        "digit_1": "accuracy",
        "digit_2": "accuracy",
        "digit_3": "accuracy",
        "digit_4": "accuracy",
        "digit_5": "accuracy",
    }
)

model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

model.save(MODEL_PATH)

with open("training_history.json", "w") as f:
    json.dump(
        {k: [float(v) for v in vals] for k, vals in history.history.items()},
        f,
        indent=2
    )

print("Saved model:", MODEL_PATH)
