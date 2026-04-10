"""
new_preprocess.py
=================
Updated preprocessing pipeline for the 82% accuracy EfficientNetB4 model.

Key changes from preprocess.py:
  - Default output size changed to 380×380 (EfficientNetB4 native)
  - Added `prepare_for_efficientnetb4()` — normalizes to [0,255] float32
    (NOT [0,1] — EfficientNetB4 uses its own internal normalization)
  - Added `preprocess_for_gradcam()` — returns both the model-ready tensor
    AND the display-ready RGB image for Grad-CAM overlay
  - Ben Graham pipeline is unchanged but resize target defaults to 380

Usage:
    from src.new_preprocess import preprocess_for_efficientnetb4, preprocess_for_gradcam
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union
import os
from tqdm import tqdm

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_SIZE_B4 = (380, 380)   # EfficientNetB4 native resolution


# ── Low-level helpers (same as preprocess.py) ─────────────────────────────────

def crop_image_from_gray(img: np.ndarray, tol: int = 7) -> np.ndarray:
    """Crop black borders from a retinal fundus image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    mask = gray > tol
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return img
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    return img[y0:y1+1, x0:x1+1]


def circle_crop(img: np.ndarray, sigmaX: int = 10) -> np.ndarray:
    """
    Circular crop — focus the model on the retina disc area by masking corners.
    Identical to preprocess.py but extracted here for direct use.
    """
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    r = min(cx, cy)

    y_c, x_c = np.ogrid[:h, :w]
    mask = (x_c - cx)**2 + (y_c - cy)**2 <= r**2

    masked = img.copy()
    masked[~mask] = 0

    blurred = cv2.GaussianBlur(masked, (0, 0), sigmaX)
    return crop_image_from_gray(blurred, tol=7)


def apply_clahe(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE contrast enhancement in LAB color space."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    return enhanced


def subtract_local_average(img: np.ndarray, kernel_size: int = 50) -> np.ndarray:
    """Subtract local Gaussian average to normalize illumination."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    local_avg = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return cv2.addWeighted(img, 4, local_avg, -4, 128)


# ── Retinal Image Validator ───────────────────────────────────────────────────

def is_retinal_image(img_bgr, threshold=0.15):
    """
    Validates whether the uploaded image is a retinal fundus photograph
    (either raw or Ben Graham preprocessed). Rejects documents, screenshots,
    and other non-retinal images.

    Returns: (is_valid, confidence, reason, state)
      state = "PREPROCESSED" | "RAW"
    """
    if img_bgr is None or img_bgr.size == 0:
        return False, 0.0, "Empty or unreadable image", "RAW"

    h, w = img_bgr.shape[:2]
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    b_ch, g_ch, r_ch = cv2.split(img_bgr)

    overall_mean = float(np.mean(gray))

    # ── 1. Detect PREPROCESSED (Ben Graham) signature first ──────────────────
    # If it matches our medical signature (gray-ish center near 128), we accept immediately.
    cy, cx  = h // 2, w // 2
    ry, rx  = int(h * 0.30), int(w * 0.30)
    center_roi = gray[cy-ry:cy+ry, cx-rx:cx+rx]
    center_mean = float(np.mean(center_roi))
    center_std  = float(np.std(center_roi))

    c_spread = max(float(np.mean(r_ch)), float(np.mean(g_ch)), float(np.mean(b_ch))) - \
               min(float(np.mean(r_ch)), float(np.mean(g_ch)), float(np.mean(b_ch)))
    
    # Range (70 to 200) to catch various preprocessed dataset styles
    all_near_128 = (70 < center_mean < 200)

    # HARDENING: Reject Digital UIs (dark mode) which are too "flat" or symmetric.
    # Biological padding (like in dataset images) has subtle noise/variation.
    q_h, q_w = h // 10, w // 10
    corner_stds = [np.std(gray[:q_h, :q_w]), np.std(gray[:q_h, -q_w:]), 
                   np.std(gray[-q_h:, :q_w]), np.std(gray[-q_h:, -q_w:])]
    is_mathematically_exact = all(s < 0.2 for s in corner_stds) # flat corners = screenshot

    if all_near_128 and c_spread < 60 and center_std > 10 and not is_mathematically_exact:
        return True, 0.99, "Preprocessed fundus image detected", "PREPROCESSED"

    # ── 2. Hardened Intelligence RAW Eye Detection ───────────────────────────
    
    # A. Red/Biological Color Gamut Check (HSV)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hue, sat, _ = cv2.split(hsv)
    # Retina hues are typically Red-Orange (0-40) or (160-180)
    eye_hue_mask = ((hue < 35) | (hue > 155)) & (sat > 40)
    eye_pixels_ratio = np.sum(eye_hue_mask) / (h * w)
    
    if eye_pixels_ratio < 0.40 and overall_mean > 40:
        if c_spread > 50:
            return False, 0.88, "Non-biological color profile (Eye hues expected)", "RAW"

    # B. Orientation Histogram check (Text vs Biology)
    # Text/Diagram lines are perfectly straight (0/90 deg). Biological vessels are curved.
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    edge_mask = mag > 15 
    if np.sum(edge_mask) > 100:
        ang_mod = ang[edge_mask] % 90
        axis_aligned = np.sum((ang_mod < 3) | (ang_mod > 87)) / len(ang_mod)
        if axis_aligned > 0.55: # Rejection threshold at 0.55 per hardened logic
            return False, 0.84, "High artificial symmetry (Looks like text or diagram)", "RAW"

    # C. Texture & Sharpness Check (Laplacian)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var > 4500: # Sharp digital edges
        return False, 0.85, "Artificial texture (High edge density detected)", "RAW"

    # D. Peak White Rejection
    white_ratio = np.sum(gray > 250) / (h * w)
    if white_ratio > 0.12:
        return False, 0.90, "Too many white pixels (Likely a document)", "RAW"

    # E. Aspect Ratio Lockdown
    aspect = max(h, w) / max(min(h, w), 1)
    if aspect > 1.55:
        return False, 0.83, "Non-standard shape (Medical scans are square-ish)", "RAW"

    # All checks passed
    return True, 0.95, "Retinal photograph accepted (RAW)", "RAW"


# ── EfficientNetB4-specific preprocessing ─────────────────────────────────────

def ben_graham_preprocessing_b4(
    image_path: Union[str, Path],
    output_size: Tuple[int, int] = DEFAULT_SIZE_B4,
    apply_clahe_flag: bool = True,
    apply_local_avg: bool = True,
    sigmaX: int = 10,
    save_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """
    Ben Graham's preprocessing pipeline tuned for EfficientNetB4 (380×380).

    Steps:
      1. Load image (BGR)
      2. Circle crop — focus on retina
      3. Subtract local average — normalize illumination
      4. Resize to 380×380
      5. Apply CLAHE — enhance contrast

    Args:
        image_path      : Path to raw fundus image
        output_size     : Target size — default (380, 380) for EfficientNetB4
        apply_clahe_flag: Apply CLAHE enhancement (recommended: True)
        apply_local_avg : Apply local average subtraction (recommended: True)
        sigmaX          : Gaussian sigma for circle crop smoothing
        save_path       : If given, save the preprocessed image here

    Returns:
        BGR numpy array of shape (380, 380, 3), dtype uint8
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    img = circle_crop(img, sigmaX=sigmaX)

    if apply_local_avg:
        img = subtract_local_average(img, kernel_size=51)

    img = cv2.resize(img, output_size, interpolation=cv2.INTER_AREA)

    if apply_clahe_flag:
        img = apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8))

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), img)

    return img


def prepare_for_efficientnetb4(
    image_path: Union[str, Path],
    output_size: Tuple[int, int] = DEFAULT_SIZE_B4,
    apply_ben_graham: bool = True,
) -> np.ndarray:
    """
    Load, optionally preprocess, and return a model-ready float32 tensor.

    CRITICAL DIFFERENCE from the old pipeline:
      - Returns pixel values in the range [0, 255] as float32
      - Does NOT divide by 255
      - EfficientNetB4 applies its own normalization internally
      - Dividing by 255 here would reduce accuracy significantly

    Args:
        image_path      : Path to fundus image
        output_size     : Target spatial size
        apply_ben_graham: Whether to apply the circle-crop + CLAHE pipeline

    Returns:
        float32 array of shape (1, H, W, 3) — ready for model.predict()
    """
    if apply_ben_graham:
        img_bgr = ben_graham_preprocessing_b4(
            image_path, output_size=output_size,
            apply_clahe_flag=True, apply_local_avg=True,
        )
    else:
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise ValueError(f"Cannot read image: {image_path}")
        img_bgr = cv2.resize(img_bgr, output_size, interpolation=cv2.INTER_AREA)

    # Convert BGR → RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Cast to float32 — DO NOT rescale! EfficientNetB4 handles it internally.
    img_f32 = img_rgb.astype(np.float32)

    return np.expand_dims(img_f32, axis=0)   # (1, 380, 380, 3)


def preprocess_for_gradcam(
    image_path: Union[str, Path],
    output_size: Tuple[int, int] = DEFAULT_SIZE_B4,
    apply_ben_graham: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare an image for BOTH model inference AND Grad-CAM overlay.

    Returns two arrays:
      * model_input : (1, H, W, 3) float32 in [0, 255] → for model.predict()
      * display_img : (H, W, 3) uint8 RGB → used as base for Grad-CAM overlay

    Args:
        image_path      : Path to fundus image
        output_size     : Target spatial size (default 380×380)
        apply_ben_graham: Apply Ben Graham pipeline before inference

    Returns:
        (model_input, display_img)
    """
    if apply_ben_graham:
        img_bgr = ben_graham_preprocessing_b4(
            image_path, output_size=output_size,
        )
    else:
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise ValueError(f"Cannot read image: {image_path}")
        img_bgr = cv2.resize(img_bgr, output_size, interpolation=cv2.INTER_AREA)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Model input: float32 [0,255]
    model_input = np.expand_dims(img_rgb.astype(np.float32), axis=0)

    # Display image: uint8 for visualization
    display_img = img_rgb.astype(np.uint8)

    return model_input, display_img


# ── Batch preprocessing ────────────────────────────────────────────────────────

def preprocess_batch_b4(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    image_ids: Optional[list] = None,
    output_size: Tuple[int, int] = DEFAULT_SIZE_B4,
    apply_clahe_flag: bool = True,
    apply_local_avg: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Batch preprocess images to 380×380 for EfficientNetB4 training.

    Identical flow to preprocess.py preprocess_batch(), but:
      - Default output size is 380×380 (not 512×512)
      - Uses the b4-tuned pipeline

    Args:
        input_dir       : Directory with raw images
        output_dir      : Directory to save preprocessed images
        image_ids       : List of image IDs (stems). If None, process all.
        output_size     : Target size
        apply_clahe_flag: Apply CLAHE
        apply_local_avg : Apply local average subtraction
        verbose         : Show progress bar

    Returns:
        Dict with 'total', 'successful', 'failed', 'failed_ids'
    """
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if image_ids is None:
        img_files = (
            list(input_dir.glob('*.png')) +
            list(input_dir.glob('*.jpg')) +
            list(input_dir.glob('*.jpeg'))
        )
        image_ids = [f.stem for f in img_files]

    stats = {
        'total'     : len(image_ids),
        'successful': 0,
        'failed'    : 0,
        'failed_ids': [],
    }

    iterator = tqdm(image_ids, desc=f"Preprocessing → {output_size[0]}px") \
               if verbose else image_ids

    for img_id in iterator:
        out_path = output_dir / f"{img_id}.png"
        if out_path.exists():
            stats['successful'] += 1
            continue

        # Find input path
        in_path = None
        for ext in ('.png', '.jpg', '.jpeg'):
            candidate = input_dir / f"{img_id}{ext}"
            if candidate.exists():
                in_path = candidate
                break

        if in_path is None:
            stats['failed'] += 1
            stats['failed_ids'].append(img_id)
            continue

        try:
            ben_graham_preprocessing_b4(
                image_path=in_path,
                output_size=output_size,
                apply_clahe_flag=apply_clahe_flag,
                apply_local_avg=apply_local_avg,
                save_path=out_path,
            )
            stats['successful'] += 1
        except Exception as e:
            if verbose:
                print(f"  Error processing {img_id}: {e}")
            stats['failed'] += 1
            stats['failed_ids'].append(img_id)

    if verbose:
        print("\n" + "=" * 60)
        print("  PREPROCESSING SUMMARY  (380×380 EfficientNetB4)")
        print("=" * 60)
        print(f"  Total images   : {stats['total']}")
        print(f"  Successful     : {stats['successful']}")
        print(f"  Failed         : {stats['failed']}")
        if stats['failed'] > 0:
            print(f"  Failed IDs     : {stats['failed_ids'][:10]}")
        print("=" * 60)

    return stats


# ── Visualization helper ───────────────────────────────────────────────────────

def compare_preprocessing_b4(
    image_path: Union[str, Path],
    output_size: Tuple[int, int] = DEFAULT_SIZE_B4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return four stages of preprocessing for side-by-side visualization.

    Returns:
        (original_resized, after_crop, after_local_avg, final_clahe)
        All are RGB uint8, shape (H, W, 3)
    """
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise ValueError(f"Cannot read: {image_path}")

    def to_rgb(bgr): return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Stage 0: plain resize
    orig = to_rgb(cv2.resize(img_bgr, output_size))

    # Stage 1: circle crop
    cropped = to_rgb(cv2.resize(circle_crop(img_bgr.copy()), output_size))

    # Stage 2: + local average subtraction
    local_avg_img = subtract_local_average(
        cv2.resize(circle_crop(img_bgr.copy()), output_size)
    )
    normalized = to_rgb(local_avg_img)

    # Stage 3: + CLAHE
    final = to_rgb(apply_clahe(local_avg_img.copy()))

    return orig, cropped, normalized, final


# ── Standalone demo ────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 65)
    print("  new_preprocess.py — EfficientNetB4 Preprocessing Demo")
    print("=" * 65)

    sample = "data/raw/train_images/000c1434d8d7.png"

    if not Path(sample).exists():
        print(f"  Sample image not found: {sample}")
        print("  Update the path to test.")
        return

    print(f"\n  Input  : {sample}")

    tensor, display = preprocess_for_gradcam(image_path=sample)
    print(f"  Model tensor   : shape={tensor.shape}, dtype={tensor.dtype}")
    print(f"  Pixel range    : [{tensor.min():.1f}, {tensor.max():.1f}]  ← [0,255] for EfficientNetB4")
    print(f"  Display image  : shape={display.shape}, dtype={display.dtype}")

    out_path = "data/processed_b4/sample_380px.png"
    ben_graham_preprocessing_b4(sample, save_path=out_path)
    print(f"\n  Saved 380px preprocessed image: {out_path}")

    print("\n✅ new_preprocess.py ready for the 82% EfficientNetB4 pipeline.")
    print("=" * 65)


if __name__ == "__main__":
    main()

# ── Aliases for webapp compatibility ─────────────────────────────────────────
ben_graham_preprocessing = ben_graham_preprocessing_b4
