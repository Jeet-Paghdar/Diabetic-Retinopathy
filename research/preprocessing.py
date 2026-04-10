"""
research/preprocessing.py - Literature Preprocessing Filters
=============================================================
Contains the specific image processing pipelines for:
1. Standard [0,1] normalization (Arora 2024)
2. SOP — Circular crop + HSV Equalization (Lin & Wu 2023)
3. ImageNet Standardization (EffNet-SVM 2025)
"""

import cv2
import numpy as np

# ── 1. SOP PREPROCESSING (LIN & WU 2023) ──────────────────────────────────────

def apply_sop_preprocessing(img_bgr, target_size=(224, 224)):
    """
    SOP Methodology (Revised ResNet-50):
    1. Circular Crowing: Remove black corners from fundus images.
    2. HSV Mapping: Convert to HSV to enhance vessel contrast.
    3. Histogram Equalization: Perform CLAHE on the Value (V) channel.
    """
    # 1. Resize for processing
    img = cv2.resize(img_bgr, target_size)
    
    # 2. Circular Masking
    h, w, _ = img.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(w / 2), int(h / 2)), int(min(h, w) / 2), 255, -1)
    img = cv2.bitwise_and(img, img, mask=mask)
    
    # 3. HSV Histogram Equalization
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_c, s_c, v_c = cv2.split(hsv)
    
    # Apply CLAHE to the V channel (brightness)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_c = clahe.apply(v_c)
    
    # Merge back and convert to RGB
    hsv_final = cv2.merge([h_c, s_c, v_c])
    img_rgb = cv2.cvtColor(hsv_final, cv2.COLOR_HSV2RGB)
    
    return img_rgb / 255.0  # Normalized to [0,1]

# ── 2. STANDARD SCALING (ARORA ET AL. 2024) ───────────────────────────────────

def apply_standard_normalization(img_bgr, target_size=(224, 224)):
    """
    Arora et al. (2024): Simple resizing + RGB [0,1] scaling.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size)
    return img_resized / 255.0

# ── 3. IMAGENET STANDARDIZATION (EFFNET-SVM 2025) ─────────────────────────────

def apply_imagenet_scaling(img_bgr, target_size=(224, 224)):
    """
    EffNet-SVM (2025): Uses pre-trained V2-S which expects ImageNet stats.
    Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size) / 255.0
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img_normalized = (img_resized - mean) / std
    return img_normalized

# ── SUMMARY ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing Preprocessing Filters...")
    dummy_img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    cv2.circle(dummy_img, (500, 500), 400, (255, 100, 50), -1)
    
    p1 = apply_sop_preprocessing(dummy_img)
    p2 = apply_standard_normalization(dummy_img)
    p3 = apply_imagenet_scaling(dummy_img)
    
    print(f"SOP Shape:      {p1.shape} | Range: [{p1.min():.2f}, {p1.max():.2f}]")
    print(f"Standard Shape: {p2.shape} | Range: [{p2.min():.2f}, {p2.max():.2f}]")
    print(f"ImageNet Shape: {p3.shape} | Range: [{p3.min():.2f}, {p3.max():.2f}]")
    print("Preprocessing filters ready for scientific benchmark.")
