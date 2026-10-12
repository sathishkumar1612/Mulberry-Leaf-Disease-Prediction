"""
Mulberry Leaf Disease Classification — EfficientNetB0 Training Pipeline
=========================================================================
Features:
  - EfficientNetB0 backbone (ImageNet pretrained, 224×224)
  - Advanced data augmentation (flip, rotate, zoom, translate, contrast)
  - Cosine learning rate decay with linear warmup
  - Automatic class weight balancing from directory counts
  - Two-phase training: head warmup → backbone fine-tuning
  - Comprehensive evaluation: classification report + confusion matrix
"""

import os
import math
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report, confusion_matrix


# ---------------------------------------------------------------------------
# Learning Rate Schedule: Linear Warmup + Cosine Decay
# ---------------------------------------------------------------------------

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate that linearly warms up then follows cosine decay."""

    def __init__(self, base_lr, warmup_steps, total_steps, min_lr=1e-7):
        super().__init__()
        self.base_lr = float(base_lr)
        self.warmup_steps = float(warmup_steps)
        self.total_steps = float(total_steps)
        self.min_lr = float(min_lr)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        # Linear warmup
        warmup_lr = self.base_lr * (step / tf.maximum(self.warmup_steps, 1.0))
        # Cosine decay
        decay_progress = (step - self.warmup_steps) / tf.maximum(
            self.total_steps - self.warmup_steps, 1.0
        )
        decay_progress = tf.minimum(decay_progress, 1.0)
        cosine_lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
            1.0 + tf.math.cos(math.pi * decay_progress)
        )
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr": self.min_lr,
        }


# ---------------------------------------------------------------------------
# Model Builder
# ---------------------------------------------------------------------------

def build_efficientnet_model(input_shape=(224, 224, 3), num_classes=3):
    """
    Build an EfficientNetB0 classifier with integrated augmentation and a
    two-hidden-layer classification head with L2 regularisation.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # ── Integrated Data Augmentation (only active during training) ──
    x = layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = layers.RandomRotation(0.20)(x)
    x = layers.RandomZoom(0.20)(x)
    x = layers.RandomTranslation(0.15, 0.15)(x)
    x = layers.RandomContrast(0.15)(x)

    # EfficientNet preprocess expects [0, 255]; our pipeline feeds [0, 1]
    x = x * 255.0
    x = tf.keras.applications.efficientnet.preprocess_input(x)

    # ── Pre-trained EfficientNetB0 backbone ──
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # frozen during Phase 1

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(
        512, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(
        256, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(
        inputs=inputs, outputs=outputs,
        name="Mulberry_EfficientNetB0_Classifier",
    )
    return model, base_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _count_images(directory):
    """Return {class_name: count} for every sub-folder in *directory*."""
    counts = {}
    for cls_name in sorted(os.listdir(directory)):
        cls_path = os.path.join(directory, cls_name)
        if not os.path.isdir(cls_path):
            continue
        n = sum(
            1 for f in os.listdir(cls_path)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        )
        counts[cls_name] = n
    return counts


def compute_class_weights(train_dir):
    """Compute balanced class weights so rare classes get higher weight."""
    counts = _count_images(train_dir)
    total = sum(counts.values())
    n_classes = len(counts)
    weights = {
        i: total / (n_classes * count)
        for i, (_, count) in enumerate(counts.items())
    }
    print(f"  Class distribution : {counts}")
    print(f"  Computed weights   : { {k: round(v, 3) for k, v in weights.items()} }")
    return weights


def load_dataset(train_dir, val_dir, img_size=(224, 224), batch_size=32):
    """Load train/val datasets, rescale to [0,1], and prefetch."""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        label_mode="categorical",
        seed=42,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,
        label_mode="categorical",
    )

    rescale = layers.Rescaling(1.0 / 255)
    train_ds = train_ds.map(lambda x, y: (rescale(x), y)).prefetch(tf.data.AUTOTUNE)
    val_ds   = val_ds.map(lambda x, y: (rescale(x), y)).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------

def train(
    train_dir,
    val_dir,
    output_path="mulberry_leaf_disease_Identification_v2.keras",
    warmup_epochs=10,
    finetune_epochs=15,
    batch_size=32,
):
    print(f"📁 Training data   : {train_dir}")
    print(f"📁 Validation data : {val_dir}")

    train_ds, val_ds = load_dataset(train_dir, val_dir, batch_size=batch_size)

    # Discover class info
    raw_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=(224, 224), batch_size=1
    )
    class_names = raw_ds.class_names
    num_classes = len(class_names)
    print(f"🏷️  {num_classes} classes: {class_names}")

    class_weights = compute_class_weights(train_dir)

    # Count training images for LR schedule
    train_count = sum(_count_images(train_dir).values())
    steps_per_epoch = max(1, train_count // batch_size)

    # Build model
    model, base_model = build_efficientnet_model(
        input_shape=(224, 224, 3), num_classes=num_classes
    )
    model.summary()

    # ── Phase 1 — Warmup (backbone frozen) ──────────────────────────
    print("\n🔥 Phase 1: Training Classification Head (Backbone Frozen)…")

    total_warmup_steps = float(steps_per_epoch * warmup_epochs)
    lr_phase1 = WarmupCosineDecay(
        base_lr=1e-3,
        warmup_steps=float(steps_per_epoch * 2),
        total_steps=total_warmup_steps,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_phase1),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    phase1_cb = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=5,
            restore_best_weights=True, verbose=1,
        ),
        callbacks.ModelCheckpoint(
            output_path, monitor="val_accuracy",
            save_best_only=True, verbose=1,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=warmup_epochs,
        callbacks=phase1_cb,
        class_weight=class_weights,
    )

    # ── Phase 2 — Fine-tuning (unfreeze top backbone layers) ────────
    print("\n🔧 Phase 2: Fine-Tuning Top Backbone Layers…")
    base_model.trainable = True
    freeze_until = int(len(base_model.layers) * 0.6)
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False

    trainable = sum(1 for l in base_model.layers if l.trainable)
    print(f"   Unfroze {trainable}/{len(base_model.layers)} backbone layers")

    total_ft_steps = float(steps_per_epoch * finetune_epochs)
    lr_phase2 = WarmupCosineDecay(
        base_lr=1e-5,
        warmup_steps=float(steps_per_epoch),
        total_steps=total_ft_steps,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_phase2),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    phase2_cb = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=5,
            restore_best_weights=True, verbose=1,
        ),
        callbacks.ModelCheckpoint(
            output_path, monitor="val_accuracy",
            save_best_only=True, verbose=1,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=finetune_epochs,
        callbacks=phase2_cb,
        class_weight=class_weights,
    )

    print(f"\n✅ Model saved to: {output_path}")

    # ── Evaluation ──────────────────────────────────────────────────
    print("\n📊 Evaluation on Validation Set:")
    y_true, y_pred = [], []
    for x_batch, y_batch in val_ds:
        preds = model.predict(x_batch, verbose=0)
        y_true.extend(np.argmax(y_batch.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    print("\n─── CLASSIFICATION REPORT ───")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    print("─── CONFUSION MATRIX ───")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    print("\n─── PER-CLASS ACCURACY ───")
    for i, name in enumerate(class_names):
        if cm[i].sum() > 0:
            acc = cm[i][i] / cm[i].sum() * 100
            print(f"  {name}: {acc:.1f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train EfficientNetB0 for Mulberry Leaf Disease Classification"
    )
    parser.add_argument("--train_dir", type=str, required=True,
                        help="Path to training data directory")
    parser.add_argument("--val_dir", type=str, required=True,
                        help="Path to validation data directory")
    parser.add_argument("--output", type=str,
                        default="mulberry_leaf_disease_Identification_v2.keras")
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--finetune_epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    train(
        args.train_dir, args.val_dir, args.output,
        args.warmup_epochs, args.finetune_epochs, args.batch_size,
    )
