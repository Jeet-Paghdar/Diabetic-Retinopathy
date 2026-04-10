"""
gradcam_utils.py
----------------
RESTORED GRAD-CAM ENGINE
Strictly follows the 'Split-Execution' pattern from Notebook 11.
Bypasses 'Functional.call()' errors by manually handling nested model components.
"""

import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

def compute_gradcam(model, img_arr, pred_grade, alpha=0.45):
    """
    Computes Grad-CAM using the specific architecture from the database integration notebook.
    This version is restored to ensure stable inference in the Streamlit environment.
    """
    try:
        # 1. Locate the base (EfficientNet) sub-model
        base_model = None
        for layer in model.layers:
            if 'efficientnet' in layer.name.lower():
                base_model = layer
                break
        
        if base_model is None:
            # Recursive search fallback
            def find_submodel(m):
                for l in m.layers:
                    if 'efficientnet' in l.name.lower(): return l
                    if hasattr(l, 'layers'):
                        res = find_submodel(l)
                        if res: return res
                return None
            base_model = find_submodel(model)

        if base_model is None:
            raise ValueError("EfficientNet sub-model not found.")

        # 2. Get target conv layer (top_activation)
        try:
            conv_layer = base_model.get_layer('top_activation')
        except ValueError:
            # Fallback to last 4D activation
            conv_layer = [l for l in reversed(base_model.layers) if len(l.output_shape) == 4][0]

        # 3. Create the Gradient Model (sub-model input -> [conv_out, sub_model_out])
        # This matches the 'Working' architecture from Notebook 11.
        grad_model = keras.Model(
            inputs=base_model.input,
            outputs=[conv_layer.output, base_model.output]
        )

        # 4. Prepare intermediate input (forward pass through any pre-layers)
        img_tensor = tf.cast(img_arr, tf.float32)
        
        pre_layers = []
        for layer in model.layers:
            if layer is base_model or 'efficientnet' in layer.name.lower():
                break
            pre_layers.append(layer)

        x = img_tensor
        for lyr in pre_layers:
            try:
                x = lyr(x, training=False)
            except:
                pass
        intermediate_input = x

        # 5. Compute Gradients
        with tf.GradientTape() as tape:
            tape.watch(intermediate_input)
            conv_outputs, predictions = grad_model(intermediate_input, training=False)
            
            # Handle list predictions
            if isinstance(predictions, (list, tuple)):
                predictions = predictions[-1]
                
            # Use prediction score
            loss = predictions[:, pred_grade]

        # 6. Extract Map and Gradients
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight feature maps
        heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Apply RELU (Standard)
        heatmap = tf.nn.relu(heatmap).numpy()

        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        # 7. Post-processing: Resize and Blur
        H, W = img_arr.shape[1], img_arr.shape[2]
        heatmap_resized = cv2.resize(heatmap, (W, H))
        heatmap_resized = cv2.GaussianBlur(heatmap_resized, (7, 7), 0)
        
        # Final Normalization after blur
        if heatmap_resized.max() > 0:
            heatmap_resized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min())

        # 8. Color Overlay (COLORMAP_JET)
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        orig_bgr = cv2.cvtColor(np.clip(img_arr[0], 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(orig_bgr, 1 - alpha, heatmap_color, alpha, 0)

        return heatmap_resized, overlay

    except Exception as e:
        # Neutral backup to prevent crash
        H, W = img_arr.shape[1], img_arr.shape[2]
        dummy_h = np.zeros((H, W), dtype=np.float32)
        dummy_o = cv2.cvtColor(np.clip(img_arr[0], 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        return dummy_h, dummy_o
