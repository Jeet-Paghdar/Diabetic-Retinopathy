"""
new_data_loader.py
==================
Updated data loader for the 82% accuracy EfficientNetB4 model.
Supports combined APTOS 2019 + EyePACS datasets and the new
EfficientNetB4 input pipeline (380x380, no manual rescaling —
the model uses its own preprocessing layer).

Key differences from data_loader.py:
  - Image size: 380×380 (EfficientNetB4 native input)
  - No rescale=1./255 — EfficientNetB4 uses internal scaling
  - Oversampling for minority classes (Grades 1, 3, 4)
  - Supports both APTOS and EyePACS directory layouts
  - Integrates GradCAM result path per sample

Usage:
    from src.new_data_loader import NewRetinaScanLoader

    loader = NewRetinaScanLoader(data_dir='data/raw')
    train_df, val_df = loader.load_and_split()
    train_gen, val_gen = loader.create_generators(train_df, val_df)
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Constants ─────────────────────────────────────────────────────────────────
IMAGE_SIZE      = (380, 380)          # EfficientNetB4 native size
BATCH_SIZE      = 16
NUM_CLASSES     = 5
RANDOM_STATE    = 42

CLASS_NAMES = {
    0: 'No DR',
    1: 'Mild DR',
    2: 'Moderate DR',
    3: 'Severe DR',
    4: 'Proliferative DR'
}

# Severity levels for clinical sorting
SEVERITY = {
    0: 'Normal',
    1: 'Mild',
    2: 'Moderate',
    3: 'Severe',
    4: 'Proliferative'
}

# ── Main Loader Class ─────────────────────────────────────────────────────────

class NewRetinaScanLoader:
    """
    Data loader for the 82% EfficientNetB4 model.
    Supports APTOS 2019 and combined APTOS + EyePACS datasets.

    Improvements over the original APTOSDataLoader:
      * 380×380 input (matches EfficientNetB4 training resolution)
      * No pixel rescaling here (EfficientNetB4 includes preprocessing)
      * Built-in oversampling for rare DR grades
      * Per-sample GradCAM output path tracking
    """

    def __init__(
        self,
        data_dir: str = 'data/raw',
        image_size: Tuple[int, int] = IMAGE_SIZE,
        use_eyepacs: bool = False,
        eyepacs_dir: Optional[str] = None,
    ):
        """
        Args:
            data_dir    : Root dir for APTOS data (has train.csv, train_images/)
            image_size  : Target spatial size — default (380, 380) for EfficientNetB4
            use_eyepacs : If True, also load EyePACS samples from eyepacs_dir
            eyepacs_dir : Path to EyePACS processed images (required if use_eyepacs)
        """
        self.data_dir     = Path(data_dir)
        self.image_size   = image_size
        self.use_eyepacs  = use_eyepacs
        self.eyepacs_dir  = Path(eyepacs_dir) if eyepacs_dir else None
        self.class_names  = CLASS_NAMES

        # APTOS standard paths
        self.train_csv      = self.data_dir / 'train.csv'
        self.train_img_dir  = self.data_dir / 'train_images'

        self._verify_paths()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _verify_paths(self):
        """Validate that required files and folders exist."""
        missing = []
        for label, path in [
            ('APTOS CSV', self.train_csv),
            ('APTOS images', self.train_img_dir),
        ]:
            if not path.exists():
                missing.append(f"  {label}: {path}")

        if self.use_eyepacs:
            if self.eyepacs_dir is None:
                missing.append("  EyePACS directory: (not provided)")
            elif not self.eyepacs_dir.exists():
                missing.append(f"  EyePACS directory: {self.eyepacs_dir}")

        if missing:
            raise FileNotFoundError(
                "Missing required paths:\n" + "\n".join(missing)
            )

        print("=" * 65)
        print("  NewRetinaScanLoader — Path Verification")
        print("=" * 65)
        print(f"  APTOS CSV   : {self.train_csv}")
        print(f"  APTOS images: {self.train_img_dir}")
        print(f"  Image size  : {self.image_size[0]}×{self.image_size[1]} (EfficientNetB4)")
        if self.use_eyepacs:
            print(f"  EyePACS     : {self.eyepacs_dir}")
        print("=" * 65)

    def _build_aptos_df(self) -> pd.DataFrame:
        """Load APTOS CSV and attach image paths + metadata."""
        df = pd.read_csv(self.train_csv)
        df['image_path'] = df['id_code'].apply(
            lambda code: str(self.train_img_dir / f"{code}.png")
        )
        df['source']     = 'aptos'
        df['class_name'] = df['diagnosis'].map(CLASS_NAMES)
        df['severity']   = df['diagnosis'].map(SEVERITY)

        # Add placeholder column for GradCAM output path
        df['gradcam_path'] = None

        # Verify images exist
        df['_exists'] = df['image_path'].apply(os.path.exists)
        missing = (~df['_exists']).sum()
        if missing > 0:
            print(f"  WARNING: {missing} APTOS images not found — skipping.")
        df = df[df['_exists']].copy().drop('_exists', axis=1)

        return df

    def _build_eyepacs_df(self) -> pd.DataFrame:
        """
        Load EyePACS samples from a flat directory layout.
        Expects sub-folders named 0, 1, 2, 3, 4 (one per grade).
        """
        if not self.use_eyepacs or self.eyepacs_dir is None:
            return pd.DataFrame()

        records = []
        for grade in range(NUM_CLASSES):
            grade_dir = self.eyepacs_dir / str(grade)
            if not grade_dir.exists():
                continue
            for img_file in grade_dir.glob('*.png'):
                records.append({
                    'id_code'    : img_file.stem,
                    'diagnosis'  : grade,
                    'image_path' : str(img_file),
                    'source'     : 'eyepacs',
                    'class_name' : CLASS_NAMES[grade],
                    'severity'   : SEVERITY[grade],
                    'gradcam_path': None,
                })

        if not records:
            print("  WARNING: No EyePACS images found — check folder structure.")
        return pd.DataFrame(records)

    # ── Public API ────────────────────────────────────────────────────────────

    def load_and_split(
        self,
        val_size: float = 0.15,
        oversample_minority: bool = True,
        oversample_factor: int = 2,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load full dataset (APTOS ± EyePACS) and stratified split.

        Args:
            val_size           : Fraction for validation (default 15%)
            oversample_minority: Whether to duplicate minority-class samples in train
            oversample_factor  : Multiplier for Grades 1, 3, 4 (default 2×)

        Returns:
            (train_df, val_df) DataFrames with 'image_path', 'diagnosis', etc.
        """
        print("\n[1/3] Loading data...")
        aptos_df = self._build_aptos_df()
        print(f"  APTOS samples loaded: {len(aptos_df)}")

        if self.use_eyepacs:
            eyepacs_df = self._build_eyepacs_df()
            print(f"  EyePACS samples loaded: {len(eyepacs_df)}")
            combined_df = pd.concat([aptos_df, eyepacs_df], ignore_index=True)
        else:
            combined_df = aptos_df

        print(f"  Total samples: {len(combined_df)}")

        # ── Class distribution ────────────────────────────────────────────────
        print("\n[2/3] Class distribution (before split):")
        print("-" * 55)
        for grade in range(NUM_CLASSES):
            n = (combined_df['diagnosis'] == grade).sum()
            pct = n / len(combined_df) * 100
            print(f"  Grade {grade} ({CLASS_NAMES[grade]:18s}): {n:5d}  ({pct:5.1f}%)")
        print("-" * 55)

        # ── Stratified split ──────────────────────────────────────────────────
        print("\n[3/3] Stratified train/val split...")
        train_df, val_df = train_test_split(
            combined_df,
            test_size=val_size,
            stratify=combined_df['diagnosis'],
            random_state=RANDOM_STATE,
        )
        train_df = train_df.reset_index(drop=True)
        val_df   = val_df.reset_index(drop=True)

        # ── Oversampling ──────────────────────────────────────────────────────
        if oversample_minority:
            minority_grades = [1, 3, 4]   # Rare DR grades
            oversample_parts = []
            for grade in minority_grades:
                subset = train_df[train_df['diagnosis'] == grade]
                for _ in range(oversample_factor - 1):
                    oversample_parts.append(subset)

            if oversample_parts:
                extra_df = pd.concat(oversample_parts, ignore_index=True)
                train_df = pd.concat([train_df, extra_df], ignore_index=True)
                train_df = train_df.sample(
                    frac=1, random_state=RANDOM_STATE
                ).reset_index(drop=True)
                print(f"  After oversampling: {len(train_df)} train samples")

        print(f"\n  Final split:")
        print(f"    Train : {len(train_df)} samples")
        print(f"    Val   : {len(val_df)} samples")

        return train_df, val_df

    def create_generators(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        batch_size: int = BATCH_SIZE,
    ):
        """
        Build Keras ImageDataGenerators for EfficientNetB4.

        CRITICAL: No rescale=1./255 here — EfficientNetB4 was trained
        with pixel values in [0, 255] and uses its own internal
        preprocessing. Applying manual rescaling would hurt accuracy.

        Args:
            train_df  : Training DataFrame (must have 'image_path', 'diagnosis')
            val_df    : Validation DataFrame
            batch_size: Mini-batch size (default 16 for 380px images on GPU)

        Returns:
            (train_generator, val_generator) Keras generator tuples
        """
        # ── Training augmentation (no rescale!) ───────────────────────────────
        train_datagen = ImageDataGenerator(
            # NO rescale — EfficientNetB4 preprocessing is internal
            rotation_range=15,
            width_shift_range=0.05,
            height_shift_range=0.05,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.05,
            brightness_range=[0.85, 1.15],
            fill_mode='constant',
            cval=0,
        )

        # ── Validation: no augmentation, no rescale ───────────────────────────
        val_datagen = ImageDataGenerator()

        # Convert label column to string (required by flow_from_dataframe)
        train_df = train_df.copy()
        val_df   = val_df.copy()
        train_df['label'] = train_df['diagnosis'].astype(str)
        val_df['label']   = val_df['diagnosis'].astype(str)

        target_size = (self.image_size[0], self.image_size[1])

        train_gen = train_datagen.flow_from_dataframe(
            dataframe=train_df,
            x_col='image_path',
            y_col='label',
            target_size=target_size,
            color_mode='rgb',
            class_mode='categorical',
            batch_size=batch_size,
            shuffle=True,
            seed=RANDOM_STATE,
        )

        val_gen = val_datagen.flow_from_dataframe(
            dataframe=val_df,
            x_col='image_path',
            y_col='label',
            target_size=target_size,
            color_mode='rgb',
            class_mode='categorical',
            batch_size=batch_size,
            shuffle=False,
        )

        print(f"\n  Generators ready:")
        print(f"    Input size  : {target_size[0]}×{target_size[1]} × 3")
        print(f"    Batch size  : {batch_size}")
        print(f"    Train steps : {len(train_gen)}")
        print(f"    Val steps   : {len(val_gen)}")
        print(f"    Rescaling   : DISABLED (EfficientNetB4 preprocessing)")

        return train_gen, val_gen

    def attach_gradcam_paths(
        self,
        df: pd.DataFrame,
        gradcam_output_dir: str = 'gradcam_outputs_new',
    ) -> pd.DataFrame:
        """
        Populate the 'gradcam_path' column for each row.
        Paths follow the pattern: <output_dir>/<id_code>_gradcam.png

        Args:
            df               : DataFrame with 'id_code' column
            gradcam_output_dir: Base directory where GradCAM PNGs are saved

        Returns:
            DataFrame with 'gradcam_path' filled in
        """
        base = Path(gradcam_output_dir)
        df = df.copy()
        df['gradcam_path'] = df['id_code'].apply(
            lambda code: str(base / f"{code}_gradcam.png")
        )
        return df

    def get_class_weights(self, train_df: pd.DataFrame) -> Dict[int, float]:
        """
        Compute sklearn-style balanced class weights for use in model.fit().

        Args:
            train_df: Training DataFrame (after oversampling if desired)

        Returns:
            Dict mapping grade → weight (float)
        """
        from sklearn.utils.class_weight import compute_class_weight

        classes = np.unique(train_df['diagnosis'])
        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=train_df['diagnosis'].values,
        )
        cw = dict(zip(classes.tolist(), weights.tolist()))

        print("\n  Class Weights (for imbalance handling):")
        print("-" * 45)
        for cls in sorted(cw):
            print(f"    Grade {cls} ({CLASS_NAMES[cls]:18s}): {cw[cls]:.4f}")
        print("-" * 45)

        return cw

    def load_single_image_for_inference(
        self,
        image_path: Union[str, Path],
        apply_preprocessing: bool = True,
    ) -> np.ndarray:
        """
        Load and prepare a single image for EfficientNetB4 inference.

        IMPORTANT: Returns pixel values in [0, 255] (uint8 range as float32)
        because EfficientNetB4 applies its own normalization internally.

        Args:
            image_path         : Path to retinal fundus image
            apply_preprocessing: If True, apply Ben Graham preprocessing

        Returns:
            float32 array of shape (1, 380, 380, 3) ready for model.predict()
        """
        import cv2

        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if apply_preprocessing:
            # Minimal inline Ben Graham crop (optional if image already preprocessed)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            mask = gray > 7
            coords = np.argwhere(mask)
            if len(coords):
                y0, x0 = coords.min(axis=0)
                y1, x1 = coords.max(axis=0)
                img = img[y0:y1+1, x0:x1+1]

        # Resize to EfficientNetB4 native size
        img = cv2.resize(img, (self.image_size[1], self.image_size[0]))

        # Cast to float32 — do NOT divide by 255
        img = img.astype(np.float32)

        return np.expand_dims(img, axis=0)   # (1, 380, 380, 3)

    def print_summary(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """Pretty-print dataset summary for a notebook cell."""
        total = len(train_df) + len(val_df)

        print("\n" + "=" * 65)
        print("  NewRetinaScanLoader — Dataset Summary")
        print("=" * 65)
        print(f"  Model input size : {self.image_size[0]}×{self.image_size[1]} (EfficientNetB4)")
        print(f"  Total samples    : {total}")
        print(f"  Training samples : {len(train_df)}")
        print(f"  Validation samples: {len(val_df)}")
        print(f"  Number of classes: {NUM_CLASSES}")
        print("-" * 65)
        print(f"  {'Grade':<8} {'Name':<20} {'Train':>8} {'Val':>8}")
        print("-" * 65)
        for grade in range(NUM_CLASSES):
            tr = (train_df['diagnosis'] == grade).sum()
            vl = (val_df['diagnosis'] == grade).sum()
            print(f"  {grade:<8} {CLASS_NAMES[grade]:<20} {tr:>8} {vl:>8}")
        print("=" * 65)


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "=" * 65)
    print("  new_data_loader.py — Self-test")
    print("=" * 65)

    loader = NewRetinaScanLoader(data_dir='data/raw')
    train_df, val_df = loader.load_and_split(oversample_minority=True)
    cw = loader.get_class_weights(train_df)
    train_gen, val_gen = loader.create_generators(train_df, val_df)
    loader.print_summary(train_df, val_df)

    print("\n✅ new_data_loader.py is ready for the 82% EfficientNetB4 pipeline.")
