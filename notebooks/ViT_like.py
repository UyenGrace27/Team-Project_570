import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# =========================================================
# 1. Paths
# =========================================================
train_dir = "/Users/uyennguyen/MA_DAAN_2025_2026 /DAAN 570/Team Project_570/Data/train"   
test_dir = "/Users/uyennguyen/MA_DAAN_2025_2026 /DAAN 570/Team Project_570/Data/test"     

artifacts_dir = "artifacts"
os.makedirs(artifacts_dir, exist_ok=True)

# =========================================================
# 2. Settings
# =========================================================
IMG_SIZE = (180, 180)
BATCH_SIZE = 32
SEED = 123
NUM_CLASSES = 7

INITIAL_EPOCHS = 30
FINE_TUNE_EPOCHS = 70
TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS

UNFREEZE_LAST_N = 50

# =========================================================
# 3. Load datasets
# =========================================================
train_data = keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="int"
)

val_data = keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="int"
)

test_data = keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="int",
    shuffle=False
)

class_names = train_data.class_names
num_classes = len(class_names)

print("Class names:", class_names)
print("Number of classes:", num_classes)

if num_classes != NUM_CLASSES:
    NUM_CLASSES = num_classes

# =========================================================
# 4. Class distribution
# =========================================================
train_labels = np.concatenate([y.numpy() for _, y in train_data], axis=0)

class_counts = {class_names[i]: int(np.sum(train_labels == i)) for i in range(num_classes)}

with open(os.path.join(artifacts_dir, "class_distribution.json"), "w") as f:
    json.dump(class_counts, f, indent=4)

# =========================================================
# 5. Class weights
# =========================================================
class_indices = np.arange(num_classes)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=class_indices,
    y=train_labels
)

class_weights = {i: float(w) for i, w in enumerate(class_weights_array)}

with open(os.path.join(artifacts_dir, "class_weights.json"), "w") as f:
    json.dump({class_names[i]: float(class_weights[i]) for i in range(num_classes)}, f, indent=4)

# =========================================================
# 6. Pipeline
# =========================================================
AUTOTUNE = tf.data.AUTOTUNE
train_data = train_data.prefetch(AUTOTUNE)
val_data = val_data.prefetch(AUTOTUNE)
test_data = test_data.prefetch(AUTOTUNE)

# =========================================================
# 7. Augmentation
# =========================================================
data_augmentation = keras.Sequential([
    layers.Resizing(200, 200),
    layers.RandomCrop(180, 180),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1, fill_mode="reflect"),
    layers.RandomTranslation(0.05, 0.05, fill_mode="reflect"),
    layers.RandomZoom((-0.05, 0.05), (-0.05, 0.05), fill_mode="reflect"),
    layers.RandomContrast(0.05),
])

# =========================================================
# 8. Model
# =========================================================
def build_model(input_shape=(180, 180, 1), num_classes=7):
    inputs = layers.Input(shape=input_shape)

    x = data_augmentation(inputs)
    x = layers.Lambda(lambda img: tf.image.grayscale_to_rgb(img))(x)
    x = layers.Lambda(preprocess_input)(x)

    base_model = InceptionResNetV2(
        include_top=False,
        weights="imagenet",
        input_tensor=x
    )
    base_model.trainable = False

    y = base_model.output
    y = layers.GlobalAveragePooling2D()(y)
    y = layers.Dropout(0.4)(y)
    y = layers.Dense(128, activation="relu")(y)
    y = layers.Dropout(0.3)(y)
    outputs = layers.Dense(num_classes, activation="softmax")(y)

    return Model(inputs, outputs), base_model

model, base_model = build_model((180,180,1), num_classes)

# =========================================================
# 🔥 ADD LR SCHEDULER (ONLY ADDITION)
# =========================================================
lr_scheduler = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.75,
    patience=15,
    min_lr=1e-6,
    verbose=1
)

# =========================================================
# 9. Phase 1
# =========================================================
model.compile(
    optimizer=keras.optimizers.AdamW(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

checkpoint_path = os.path.join(artifacts_dir, "best.keras")
csv_log_path = os.path.join(artifacts_dir, "log.csv")

callbacks_phase1 = [
    keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True, mode="max"),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
    keras.callbacks.CSVLogger(csv_log_path, append=False),
    lr_scheduler
]

history1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=INITIAL_EPOCHS,
    callbacks=callbacks_phase1,
    class_weight=class_weights
)

# =========================================================
# 10. Phase 2
# =========================================================
base_model.trainable = True

for layer in base_model.layers[:-UNFREEZE_LAST_N]:
    layer.trainable = False

for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_phase2 = [
    keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True, mode="max"),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
    keras.callbacks.CSVLogger(csv_log_path, append=True),
    lr_scheduler
]

history2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=TOTAL_EPOCHS,
    initial_epoch=len(history1.history["loss"]),
    callbacks=callbacks_phase2,
    class_weight=class_weights
)

# =========================================================
# 11. Save
# =========================================================
model.save(os.path.join(artifacts_dir, "final.keras"))

# =========================================================
# 12. Evaluate
# =========================================================
test_loss, test_accuracy = model.evaluate(test_data)

print("Test accuracy:", test_accuracy)

# =========================================================
# 13. Report
# =========================================================
y_true = np.concatenate([y.numpy() for _, y in test_data], axis=0)
y_pred = np.argmax(model.predict(test_data), axis=1)

report = classification_report(y_true, y_pred, target_names=class_names)
print(report)
