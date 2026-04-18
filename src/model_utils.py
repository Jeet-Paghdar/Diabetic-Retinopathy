"""
new_model_utils.py
==================
Updated model utilities for the 82% accuracy EfficientNetB4 model.
Replaces the original EfficientNetB3 with EfficientNetB4 (380×380),
and wires in Grad-CAM generation alongside prediction.

Key differences from model_utils.py:
  - Uses EfficientNetB4 (not B3) — larger, better for fine-grained DR grading
  - Input size 380×380 (EfficientNetB4's native resolution)
  - Focal Loss for class imbalance
  - Cosine Decay LR schedule (not ReduceLROnPlateau)
  - Integrated run_gradcam() helper — returns heatmap + overlay
  - save_prediction_with_gradcam() — writes DB-ready result dict

Usage:
    from src.new_model_utils import build_efficientnetb4, run_gradcam
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, TensorBoard
)
from typing import Tuple

# ── Constants ─────────────────────────────────────────────────────────────────
INPUT_SHAPE  = (380, 380, 3)
NUM_CLASSES  = 5
BATCH_SIZE   = 16

CLASS_NAMES = [
    'No DR',
    'Mild DR',
    'Moderate DR',
    'Severe DR',
    'Proliferative DR',
]

RISK_LEVELS = {
    0: ('Low',      '#27ae60'),
    1: ('Low-Med',  '#f1c40f'),
    2: ('Medium',   '#e67e22'),
    3: ('High',     '#e74c3c'),
    4: ('Critical', '#8e44ad'),
}

GRADCAM_LAYER = 'top_activation'   # Last conv activation in EfficientNetB4


# ─────────────────────────────────────────────────────────────────────────────
# 1. MODEL BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_efficientnetb4(
    input_shape: tuple = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
    dropout_rate: float = 0.4,
    fine_tune_layers: int = 0,
) -> keras.Model:
    """
    Build EfficientNetB4 model pre-trained on ImageNet.

    Training strategy (two-phase, mirroring the 82% notebook):
      Phase 1 → Freeze base, train head only
      Phase 2 → Unfreeze top `fine_tune_layers` layers

    Args:
        input_shape      : (H, W, C) — default (380, 380, 3)
        num_classes      : Output classes — 5 for DR grading
        dropout_rate     : Dropout before final Dense
        fine_tune_layers : If > 0, unfreeze the last N layers of the base

    Returns:
        Uncompiled Keras Model  (call compile_model_b4 separately)
    """
    # Base model — ImageNet pretrained, no top
    base = EfficientNetB4(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape,
    )

    # Phase 1: freeze everything
    base.trainable = False

    # Phase 2: selectively unfreeze
    if fine_tune_layers > 0:
        for layer in base.layers[:-fine_tune_layers]:
            layer.trainable = False
        for layer in base.layers[-fine_tune_layers:]:
            layer.trainable = not isinstance(
                layer, (layers.BatchNormalization,)
            )
        print(f"  Fine-tune: last {fine_tune_layers} base layers unfrozen")
        print(f"  BatchNorm layers kept frozen (best practice)")
    else:
        print(f"  Phase 1: all base layers frozen")

    # Custom classification head
    inputs = keras.Input(shape=input_shape, name='retina_input')
    x = base(inputs, training=False)                        # (B, 12, 12, 1792)
    x = layers.GlobalAveragePooling2D(name='gap')(x)        # (B, 1792)
    x = layers.BatchNormalization(name='head_bn1')(x)
    x = layers.Dropout(dropout_rate, name='head_drop1')(x)
    x = layers.Dense(512, name='head_dense1')(x)
    x = layers.Activation('relu', name='head_relu1')(x)
    x = layers.BatchNormalization(name='head_bn2')(x)
    x = layers.Dropout(dropout_rate * 0.5, name='head_drop2')(x)
    outputs = layers.Dense(
        num_classes, activation='softmax', name='predictions'
    )(x)

    model = keras.Model(inputs, outputs, name='EfficientNetB4_DR_82pct')

    n_train = sum(tf.size(w).numpy() for w in model.trainable_weights)
    n_total = sum(tf.size(w).numpy() for w in model.weights)

    print(f"\n  EfficientNetB4 built:")
    print(f"    Total params     : {n_total:>12,}")
    print(f"    Trainable params : {n_train:>12,}")
    print(f"    Input            : {input_shape}")
    print(f"    Output           : {num_classes} classes (softmax)")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# 2. FOCAL LOSS
# ─────────────────────────────────────────────────────────────────────────────

def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """
    Focal Loss for addressing class imbalance in DR grading.
    Down-weights easy examples so the model focuses on hard cases.

    Args:
        gamma: Focusing parameter (2.0 is standard)
        alpha: Weighting factor for rare classes

    Returns:
        Loss function compatible with model.compile()
    """
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-8, 1.0 - 1e-8)
        ce     = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.math.pow(1.0 - y_pred, gamma)
        return tf.reduce_mean(tf.reduce_sum(weight * ce, axis=-1))

    loss_fn.__name__ = f'focal_loss_g{gamma}_a{alpha}'
    return loss_fn


# ─────────────────────────────────────────────────────────────────────────────
# 3. COMPILE
# ─────────────────────────────────────────────────────────────────────────────

def compile_model_b4(
    model: keras.Model,
    learning_rate: float = 1e-3,
    use_focal_loss: bool = True,
) -> keras.Model:
    """
    Compile EfficientNetB4 model with Adam + Focal Loss (or cross-entropy).

    Args:
        model          : Keras model (built by build_efficientnetb4)
        learning_rate  : Learning rate for Adam
        use_focal_loss : Use focal loss (True) or categorical cross-entropy

    Returns:
        Compiled model
    """
    loss_fn = focal_loss(gamma=2.0, alpha=0.25) if use_focal_loss \
              else 'categorical_crossentropy'

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=[
            'accuracy',
            keras.metrics.AUC(name='auc', multi_label=False),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
        ],
    )

    loss_name = 'FocalLoss' if use_focal_loss else 'CategoricalCrossEntropy'
    print(f"  Compiled | LR={learning_rate:.2e} | Loss={loss_name}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 4. CALLBACKS WITH COSINE DECAY
# ─────────────────────────────────────────────────────────────────────────────

def get_callbacks_b4(
    model_save_path: str,
    total_epochs: int = 30,
    patience_stop: int = 8,
    use_cosine_decay: bool = True,
):
    """
    Training callbacks matching the 82% accuracy notebook setup.

    Uses Cosine Decay LR schedule instead of ReduceLROnPlateau for
    smoother convergence on EfficientNetB4.

    Args:
        model_save_path : Where to save the best model (.keras)
        total_epochs    : Total training epochs (for cosine schedule)
        patience_stop   : EarlyStopping patience
        use_cosine_decay: True = cosine schedule, False = no LR schedule

    Returns:
        List of Keras callbacks
    """
    cb_list = [
        ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1,
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=patience_stop,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    if use_cosine_decay:
        def cosine_scheduler(epoch, lr):
            """Linear warmup (3 epochs) then cosine decay."""
            warmup = 3
            if epoch < warmup:
                return lr * ((epoch + 1) / warmup)
            progress = (epoch - warmup) / max(1, total_epochs - warmup)
            return lr * 0.5 * (1.0 + np.cos(np.pi * progress))

        cb_list.append(
            keras.callbacks.LearningRateScheduler(cosine_scheduler, verbose=0)
        )
        print(f"  LR schedule: Cosine Decay with {3}-epoch warmup")

    print(f"  Callbacks configured:")
    print(f"    Save path     : {model_save_path}")
    print(f"    Early stop    : patience={patience_stop}")
    return cb_list


# ─────────────────────────────────────────────────────────────────────────────
# 5. GRAD-CAM ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def run_gradcam(
    model: keras.Model,
    img_array: np.ndarray,
    pred_class: int,
    last_conv_layer_name: str = GRADCAM_LAYER,
    alpha: float = 0.45,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Grad-CAM heatmap and overlay for a single image.

    This implementation correctly handles the EfficientNetB4 sub-model
    architecture by building a dedicated grad_model that exposes the
    last convolutional activation BEFORE GlobalAveragePooling.

    Args:
        model               : Loaded EfficientNetB4 Keras Model
        img_array           : Input array, shape (1, H, W, 3), float32 [0..255]
        pred_class          : The predicted class index to visualize
        last_conv_layer_name: Name of last conv layer inside EfficientNetB4
        alpha               : Heatmap blending strength (0=original, 1=heatmap)

    Returns:
        heatmap : Raw normalized heatmap (H, W) float32
        overlay : BGR overlay image (H, W, 3) uint8 — ready to save / display
    """
    import tensorflow as tf

    # ── Step 1: Build gradient model ─────────────────────────────────────────
    # Locate the EfficientNetB4 sub-model inside the wrapper model
    base_model = None
    for layer in model.layers:
        if isinstance(layer, keras.Model) and 'efficientnet' in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        raise ValueError(
            "Could not find EfficientNetB4 sub-model inside the wrapper model. "
            "Make sure you are using build_efficientnetb4() from new_model_utils."
        )

    # Get the target conv layer output from the base model
    try:
        conv_layer = base_model.get_layer(last_conv_layer_name)
    except ValueError:
        # Fallback: use the last layer that has 4D output
        conv_layer = None
        for layer in reversed(base_model.layers):
            if len(layer.output_shape) == 4:
                conv_layer = layer
                break
        if conv_layer is None:
            raise ValueError("No suitable convolutional layer found.")
        print(f"  [GradCAM] Fallback conv layer: {conv_layer.name}")

    # Build a model: input → [conv_output, final_predictions]
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[conv_layer.output, model.output],
    )

    # ── Step 2: Compute gradients ─────────────────────────────────────────────
    img_tensor = tf.cast(img_array, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        conv_outputs, predictions = grad_model(img_tensor, training=False)
        loss = predictions[:, pred_class]

    grads = tape.gradient(loss, conv_outputs)          # (1, h, w, C)

    # ── Step 3: Pool gradients over spatial dims ──────────────────────────────
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))   # (C,)

    # ── Step 4: Weight conv feature maps by pooled grads ─────────────────────
    conv_outputs = conv_outputs[0]                     # (h, w, C)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]   # (h, w, 1)
    heatmap = tf.squeeze(heatmap)                      # (h, w)

    # ReLU + normalize to [0, 1]
    heatmap = tf.nn.relu(heatmap).numpy()
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    # ── Step 5: Resize heatmap to original image size ────────────────────────
    H, W = img_array.shape[1], img_array.shape[2]
    heatmap_resized = cv2.resize(heatmap, (W, H))

    # ── Step 6: Colormap → overlay ────────────────────────────────────────────
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    jet_colors    = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Original image in BGR uint8
    orig_bgr = cv2.cvtColor(
        np.clip(img_array[0], 0, 255).astype(np.uint8),
        cv2.COLOR_RGB2BGR,
    )

    overlay = cv2.addWeighted(orig_bgr, 1 - alpha, jet_colors, alpha, 0)

    return heatmap_resized, overlay




# ─────────────────────────────────────────────────────────────────────────────
# 6. PREDICTION + GRADCAM IN ONE CALL
# ─────────────────────────────────────────────────────────────────────────────

def predict_with_gradcam(
    model: keras.Model,
    img_array: np.ndarray,
    output_path: str = None,
    last_conv_layer: str = GRADCAM_LAYER,
    alpha: float = 0.45,
) -> dict:
    """
    Run prediction AND Grad-CAM in a single call.
    Returns a structured result dict ready for database insertion.

    Args:
        model          : Loaded & compiled EfficientNetB4 model
        img_array      : Preprocessed image (1, 380, 380, 3) float32
        output_path    : If given, save heatmap overlay PNG to this path
        last_conv_layer: Conv layer name for Grad-CAM
        alpha          : Heatmap overlay blending factor

    Returns:
        result dict with keys:
            grade, grade_name, confidence, all_probabilities,
            risk_level, risk_color, gradcam_saved_path
    """
    # ── Predict ───────────────────────────────────────────────────────────────
    probs    = model.predict(img_array, verbose=0)[0]   # (5,)
    grade    = int(np.argmax(probs))
    conf     = float(probs[grade])

    risk_label, risk_color = RISK_LEVELS[grade]

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    heatmap, overlay = run_gradcam(
        model=model,
        img_array=img_array,
        pred_class=grade,
        last_conv_layer_name=last_conv_layer,
        alpha=alpha,
    )

    saved_path = None
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cv2.imwrite(output_path, overlay)
        saved_path = output_path
        print(f"  GradCAM overlay saved: {output_path}")

    return {
        'grade'            : grade,
        'grade_name'       : CLASS_NAMES[grade],
        'confidence'       : conf,
        'all_probabilities': probs.tolist(),
        'risk_level'       : risk_label,
        'risk_color'       : risk_color,
        'heatmap'          : heatmap,
        'overlay'          : overlay,
        'gradcam_saved_path': saved_path,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_b4(
    model: keras.Model,
    val_gen,
    num_steps: int = None,
) -> dict:
    """
    Full evaluation of the EfficientNetB4 model.

    Args:
        model    : Trained model
        val_gen  : Validation generator (no shuffle)
        num_steps: Steps to evaluate (defaults to len(val_gen))

    Returns:
        Dict with: accuracy, kappa, f1, precision, recall, confusion_matrix
    """
    from sklearn.metrics import (
        accuracy_score, cohen_kappa_score,
        f1_score, precision_score, recall_score,
        confusion_matrix, classification_report,
    )

    if num_steps is None:
        num_steps = len(val_gen)

    y_true, y_pred_probs = [], []
    for _ in range(num_steps):
        X, y = next(val_gen)
        preds = model.predict(X, verbose=0)
        y_true.extend(np.argmax(y, axis=1))
        y_pred_probs.extend(preds)

    y_true       = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred       = np.argmax(y_pred_probs, axis=1)

    acc   = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    f1    = f1_score(y_true, y_pred, average='weighted')
    prec  = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec   = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    cm    = confusion_matrix(y_true, y_pred)

    print("\n" + "=" * 65)
    print("  EfficientNetB4 (82%) — Evaluation Results")
    print("=" * 65)
    print(f"  Accuracy              : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Quadratic Kappa (QWK) : {kappa:.4f}")
    print(f"  Weighted F1           : {f1:.4f}")
    print(f"  Weighted Precision    : {prec:.4f}")
    print(f"  Weighted Recall       : {rec:.4f}")
    print("=" * 65)
    print("\n  Per-class Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    return {
        'accuracy' : acc, 'kappa': kappa, 'f1': f1,
        'precision': prec, 'recall': rec,
        'confusion_matrix': cm,
        'y_true'   : y_true, 'y_pred': y_pred,
        'y_pred_probs': y_pred_probs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 65)
    print("  new_model_utils.py — Sanity Check")
    print("=" * 65)

    print("\n[1] Building EfficientNetB4 (Phase 1 — frozen base)...")
    model = build_efficientnetb4(INPUT_SHAPE, NUM_CLASSES, dropout_rate=0.4)
    model = compile_model_b4(model, learning_rate=1e-3, use_focal_loss=True)

    print("\n[2] Forward pass test...")
    dummy = np.random.randint(0, 255, (2, 380, 380, 3)).astype(np.float32)
    out   = model.predict(dummy, verbose=0)
    print(f"  Output shape     : {out.shape}")
    print(f"  Softmax sums     : {out.sum(axis=1).round(4)}")

    print("\n[3] Callbacks test...")
    cbs = get_callbacks_b4('models/efficientnetb4_test.keras', total_epochs=30)
    print(f"  Callbacks count  : {len(cbs)}")

    print("\n✅ new_model_utils.py ready for the 82% EfficientNetB4 pipeline.")
    print("=" * 65)