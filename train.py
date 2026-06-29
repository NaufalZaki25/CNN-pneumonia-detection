"""
=============================================================
  PNEUMONIA DETECTION - CNN TRAINING SCRIPT
  Dataset : Kaggle "Chest X-Ray Images (Pneumonia)"
  Model   : MobileNetV2 (Transfer Learning) + Custom Head
  Author  : Your Name
=============================================================

HOW TRANSFER LEARNING WORKS (Beginner Explanation)
---------------------------------------------------
Instead of building a CNN from scratch (which needs millions of
images and hours of training), we borrow a model that was already
trained on 1.2 million everyday images (ImageNet).

That pre-trained model already "knows" how to detect:
  - Edges, textures, shapes, patterns
We then attach our own small classifier on top that learns to
distinguish NORMAL lungs from PNEUMONIA lungs.

This means we get great accuracy with far less data and time!

DATASET STRUCTURE EXPECTED
---------------------------
chest_xray/
    train/
        NORMAL/       <- images of healthy lungs
        PNEUMONIA/    <- images with pneumonia
    val/
        NORMAL/
        PNEUMONIA/
    test/
        NORMAL/
        PNEUMONIA/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ─────────────────────────────────────────────
# 1. CONFIGURATION  (edit these paths/settings)
# ─────────────────────────────────────────────

DATASET_DIR   = r"F:\Lock In\Skripsi\Script\chest_xray"          # <-- path to your Kaggle dataset folder
MODEL_SAVE    = r"F:\Lock In\Skripsi\Script\pneumonia_model.keras"  # where the trained model will be saved
IMG_SIZE      = (224, 224)              # MobileNetV2 expects 224×224 pixels
BATCH_SIZE    = 32                      # images processed at once (lower if RAM issues)
EPOCHS_FROZEN = 10                      # training with base model frozen
EPOCHS_FINE   = 10                      # fine-tuning (optional extra training)
LEARNING_RATE = 1e-4

# ─────────────────────────────────────────────
# 2. DATA LOADING & AUGMENTATION
# ─────────────────────────────────────────────
# "Augmentation" = creating slightly modified copies of your training
# images (flipped, rotated, zoomed) so the model becomes more robust
# and less likely to memorize the training data (overfitting).

print("📂 Loading dataset...")

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,          # normalize pixels from [0-255] → [0-1]
    validation_split=0.2,       # If val/ folder is too small, use part of training data for validation
    rotation_range=15,          # random rotation up to 15°
    width_shift_range=0.1,      # shift image left/right by 10%
    height_shift_range=0.1,     # shift image up/down by 10%
    shear_range=0.1,            # slight shearing distortion
    zoom_range=0.1,             # random zoom in/out
    horizontal_flip=True,       # randomly mirror the image
    fill_mode="nearest",        # fill gaps with nearest pixel
)

# Validation and Test images are NOT augmented — only rescaled
val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",        # binary = 2 classes (NORMAL vs PNEUMONIA)
    color_mode="rgb",
)

val_gen = val_test_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    color_mode="rgb",
)

test_gen = val_test_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    color_mode="rgb",
    shuffle=False,              # keep order for evaluation
)

print(f"✅ Classes: {train_gen.class_indices}")   # {'NORMAL': 0, 'PNEUMONIA': 1}
print(f"✅ Training samples  : {train_gen.samples}")
print(f"✅ Validation samples: {val_gen.samples}")
print(f"✅ Test samples      : {test_gen.samples}")

# ─────────────────────────────────────────────
# 3. HANDLE CLASS IMBALANCE
# ─────────────────────────────────────────────
# The Kaggle dataset has ~3x more PNEUMONIA images than NORMAL.
# If we ignore this, the model learns to always predict PNEUMONIA.
# "Class weights" penalize the model more for misclassifying the
# minority class (NORMAL), forcing it to pay equal attention.

total   = train_gen.samples
n_normal    = train_gen.classes.tolist().count(0)
n_pneumonia = train_gen.classes.tolist().count(1)

weight_normal    = total / (2 * n_normal)
weight_pneumonia = total / (2 * n_pneumonia)

class_weights = {0: weight_normal, 1: weight_pneumonia}
print(f"\n⚖️  Class weights → NORMAL: {weight_normal:.2f}, PNEUMONIA: {weight_pneumonia:.2f}")

# ─────────────────────────────────────────────
# 4. BUILD THE MODEL
# ─────────────────────────────────────────────

print("\n🧠 Building model...")

# --- BASE: MobileNetV2 pre-trained on ImageNet ---
# include_top=False  → remove the original classification layer
# weights="imagenet" → use pre-trained weights
base_model = MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False   # freeze base model (don't change its weights yet)

# --- CUSTOM CLASSIFICATION HEAD ---
inputs  = keras.Input(shape=(*IMG_SIZE, 3))
x       = base_model(inputs, training=False)
x       = layers.GlobalAveragePooling2D()(x)   # flatten feature maps
x       = layers.BatchNormalization()(x)        # stabilize training
x       = layers.Dense(256, activation="relu")(x)
x       = layers.Dropout(0.4)(x)               # randomly disable 40% neurons (prevent overfitting)
x       = layers.Dense(128, activation="relu")(x)
x       = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)  # sigmoid → probability [0,1]

model = keras.Model(inputs, outputs)
model.summary()

# ─────────────────────────────────────────────
# 5. COMPILE & TRAIN (PHASE 1 — FROZEN BASE)
# ─────────────────────────────────────────────

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="binary_crossentropy",  # standard loss for binary classification
    metrics=[
        "accuracy",
        keras.metrics.AUC(name="auc"),          # area under ROC curve
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
    ],
)

# Callbacks: helpers that run automatically during training
callbacks = [
    # Stop training if validation loss doesn't improve for 5 epochs
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
    ),
    # Save the model checkpoint whenever val_loss improves
    keras.callbacks.ModelCheckpoint(
        MODEL_SAVE, monitor="val_loss", save_best_only=True, verbose=1,
    ),
    # Reduce learning rate if val_loss plateaus
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1
    ),
]

print(f"\n🚀 Phase 1: Training with frozen base ({EPOCHS_FROZEN} epochs)...")
history1 = model.fit(
    train_gen,
    epochs=EPOCHS_FROZEN,
    validation_data=val_gen,
    class_weight=class_weights,
    callbacks=callbacks,
)

# ─────────────────────────────────────────────
# 6. FINE-TUNING (PHASE 2 — UNFREEZE LAST LAYERS)
# ─────────────────────────────────────────────
# Now we unfreeze the last 30 layers of MobileNetV2 and re-train
# with a very small learning rate. This slightly adjusts the pre-trained
# features to better fit chest X-ray images.

print(f"\n🔓 Phase 2: Fine-tuning last 30 layers ({EPOCHS_FINE} epochs)...")
base_model.trainable = True

# Unfreeze only the last 30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE / 10),  # 10x smaller LR
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name="auc"),
             keras.metrics.Precision(name="precision"),
             keras.metrics.Recall(name="recall")],
)

history2 = model.fit(
    train_gen,
    epochs=EPOCHS_FINE,
    validation_data=val_gen,
    class_weight=class_weights,
    callbacks=callbacks,
)

# ─────────────────────────────────────────────
# 7. EVALUATE ON TEST SET
# ─────────────────────────────────────────────

print("\n📊 Evaluating on test set...")
results = model.evaluate(test_gen, verbose=1)
print(f"\nTest Loss     : {results[0]:.4f}")
print(f"Test Accuracy : {results[1]*100:.2f}%")
print(f"Test AUC      : {results[2]:.4f}")
print(f"Test Precision: {results[3]:.4f}")
print(f"Test Recall   : {results[4]:.4f}")

# Detailed classification report
test_gen.reset()
y_pred_prob = model.predict(test_gen, verbose=1)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()
y_true = test_gen.classes

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["NORMAL", "PNEUMONIA"]))

# ─────────────────────────────────────────────
# 8. PLOTS — Training History & Confusion Matrix
# ─────────────────────────────────────────────

def merge_history(h1, h2):
    """Combine metrics from both training phases."""
    combined = {}
    for key in h1.history:
        combined[key] = h1.history[key] + h2.history[key]
    return combined

hist = merge_history(history1, history2)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Training Results — Pneumonia Detection", fontsize=16, fontweight="bold")

metrics = [("accuracy", "val_accuracy", "Accuracy"),
           ("loss",     "val_loss",     "Loss"),
           ("auc",      "val_auc",      "AUC"),
           ("precision","val_precision","Precision")]

for ax, (train_m, val_m, title) in zip(axes.flatten(), metrics):
    ax.plot(hist[train_m], label="Train", linewidth=2)
    ax.plot(hist[val_m],   label="Validation", linewidth=2, linestyle="--")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("training_history.png", dpi=150, bbox_inches="tight")
print("\n✅ Saved: training_history.png")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["NORMAL", "PNEUMONIA"],
            yticklabels=["NORMAL", "PNEUMONIA"])
plt.title("Confusion Matrix", fontweight="bold")
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
print("✅ Saved: confusion_matrix.png")

print(f"\n🎉 Training complete! Model saved to: {MODEL_SAVE}")
