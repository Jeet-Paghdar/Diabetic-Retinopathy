<div align="center">

<img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/TensorFlow-2.17.1-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
<img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
<img src="https://img.shields.io/badge/EfficientNetB4-380×380-00C7B7?style=for-the-badge" />
<img src="https://img.shields.io/badge/Accuracy-82.05%25-27ae60?style=for-the-badge" />
<img src="https://img.shields.io/badge/Kappa_(QWK)-0.871-8e44ad?style=for-the-badge" />

<br/><br/>

# RetinaScan AI
### Automated Diabetic Retinopathy Severity Grading

*A production-ready deep learning system for clinical-grade, 5-class DR detection from retinal fundus photographs — powered by EfficientNetB4, Grad-CAM explainability, and a hybrid MySQL/SQLite database.*

<br/>

[Project Overview](#project-overview) • [Key Features](#key-features) • [Architecture](#system-architecture) • [Results](#model-performance) • [Quick Start](#quick-start) • [Project Structure](#project-structure) • [Notebooks](#research-notebooks) • [Contributing](#contributing)

</div>

---

## Project Overview

**Diabetic Retinopathy (DR)** is the leading cause of preventable blindness worldwide, affecting over 100 million diabetics. Early detection is critical — but ophthalmologist availability is severely limited in developing regions.

**RetinaScan AI** addresses this gap by providing:

- **Automated 5-class DR severity grading** from retinal fundus images (No DR → Proliferative DR)
- **Clinical interpretability** through Grad-CAM heatmap overlays that highlight infected retinal regions
- **A production web application** built on Streamlit with patient record management
- **A hybrid database architecture** (MySQL primary + SQLite fallback) for zero-downtime reliability

The system is built on a three-phase transfer learning pipeline using **EfficientNetB4** (pretrained on ImageNet) trained on the merged **APTOS 2019 + EyePACS Kaggle 2015** datasets — the two gold-standard DR grading benchmarks.

---

## Key Features

| Feature | Description |
|---|---|
| **EfficientNetB4 Backbone** | Native 380×380 resolution, ImageNet pretrained, selected for its superior compound scaling for fine-grained medical image classification |
| **Focal Loss** | Tackles severe class imbalance (No DR cases dominate real-world datasets) by down-weighting easy negatives |
| **Grad-CAM XAI** | Per-prediction heatmaps overlaid on the original fundus image, enabling clinicians to visually verify model decisions |
| **Ben Graham Preprocessing** | Circular crop + local average subtraction + CLAHE (Contrast Limited Adaptive Histogram Equalization) — the competition-winning preprocessing strategy |
| **Streamlit Web App** | Real-time clinical dashboard with drag-and-drop image upload, instant diagnosis, probability charts, and patient record logging |
| **Hybrid Database** | MySQL primary with automatic SQLite failover — scan records, Grad-CAM paths, and model versioning all persisted |
| **Retinal Image Validator** | Multi-heuristic CV filter that rejects documents, screenshots, and non-medical images before they reach the model |
| **Three-Phase Fine-Tuning** | Progressive unfreezing strategy across three training phases with Cosine Decay LR schedules |

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     RETINASCAN AI PIPELINE                   │
└──────────────────────────────────────────────────────────────┘

  Upload                  Validate                 Preprocess
  Fundus Image    ──▶    is_retinal_image()   ──▶  Ben Graham Pipeline
  (any size)             (CV heuristics)            - Circle Crop
                                                     - Local Avg Subtract
                                                     - Resize → 380×380
                                                     - CLAHE Enhancement
                                ▼

  Inference                                         Grad-CAM
  EfficientNetB4   ──▶   5-class Softmax    ──▶   top_activation layer
  (380×380 input)        [No DR → PDR]             GradientTape
  Label smoothing        + Confidence %            Heatmap + Overlay PNG
                                ▼

  Results UI                                        Database
  - DR Grade + Name   ──▶   Streamlit        ──▶   MySQL (primary)
  - Confidence bar           Dashboard               SQLite (fallback)
  - Probability chart        Patient Form            scans table
  - Grad-CAM overlay         Risk indicator          gradcam_path stored
```

### DR Severity Classes

| Grade | Class | Risk Level | Description |
|:---:|---|:---:|---|
| 0 | No DR | Low | No visible retinal changes |
| 1 | Mild DR | Low-Medium | Microaneurysms only |
| 2 | Moderate DR | Medium | More than microaneurysms; less than severe |
| 3 | Severe DR | High | Extensive hemorrhages, venous beading |
| 4 | Proliferative DR | Critical | Neovascularization; vitreous hemorrhage risk |

---

## Model Performance

> Evaluated on the merged **APTOS 2019 + EyePACS Kaggle 2015** holdout set.

| Metric | Score |
|---|---|
| **Validation Accuracy** | **82.05%** |
| **Quadratic Weighted Kappa (QWK)** | **0.871** |
| **Weighted F1-Score** | **0.82** |
| **Training Accuracy** | **98.72%** |
| Dataset | Merged APTOS 2019 & EyePACS |
| Input Resolution | 380 × 380 px |
| Model | EfficientNetB4 Ensemble (Phase 1 + Phase 3) |
| Preprocessing | Ben Graham's Circular Crop & CLAHE |

> **Note on QWK (0.871):** Quadratic Weighted Kappa is the *official competition metric* for DR grading. A score of 0.871 is considered **strong agreement** for 5-class ordinal classification and is competitive with Kaggle competition benchmarks.

### Training Strategy

```
Phase 1   ──  Partial freeze: top layers of EfficientNetB4 unfrozen, BatchNorm kept frozen
              Train custom head (GlobalAvgPool → BN → Dropout 0.4 → Dense 512 → Dense 256 → Softmax)
              LR = CosineDecayRestarts (initial: 1e-4, decay steps: steps/epoch × 5)
              Loss: CategoricalCrossentropy with label_smoothing=0.05
              EarlyStopping on val_accuracy (patience=4)
              Saved as: efficientnet_phase1.keras

Phase 2   ──  Safe unfreeze: all backbone layers unfrozen, BatchNorm kept frozen
              Loads Phase 1 best weights as starting point
              LR = CosineDecayRestarts (initial: 1e-5, decay steps: steps/epoch × 5)
              Loss: CategoricalCrossentropy with label_smoothing=0.05
              EarlyStopping on val_accuracy (patience=6)
              Saved as: efficientnet_best.keras

Phase 3   ──  Deep fine-tuning: loads Phase 2 best weights
              LR = CosineDecayRestarts (initial: 1e-6, decay steps: steps/epoch × 5)
              Loss: CategoricalCrossentropy with label_smoothing=0.05
              EarlyStopping on val_accuracy (patience=10)
              Saved as: efficientnet_best.keras

Ensemble  ──  Average softmax outputs of Phase 1 + Phase 3 checkpoints
              Final Ensemble Accuracy: 82.05%
```

---

## Quick Start

### Prerequisites

- Python 3.11
- Conda (recommended) or pip
- MySQL 8.0+ *(optional — SQLite fallback is automatic)*

### 1. Clone the Repository

```bash
git clone https://github.com/Jeet-Paghdar/Diabetic-Retinopathy.git
cd Diabetic-Retinopathy
```

### 2. Create Environment

**Option A — Conda (Recommended)**
```bash
conda env create -f environment.yml
conda activate retinascan
```

**Option B — pip**
```bash
pip install -r requirements.txt
```

### 3. Download the Model Weights

Place the trained model file in the `models/` directory:
```
models/
└── efficientnetb4_best.keras
```

> If you are training from scratch, run the notebooks in order (see [Research Notebooks](#research-notebooks) below).

### 4. Configure the Database *(Optional)*

For MySQL support, update the credentials in `src/database.py`:
```python
DB_CONFIG = {
    'host'    : 'localhost',
    'port'    : 3306,
    'user'    : 'your_user',
    'password': 'your_password',
    'database': 'retinascan_db',
}
```

The app will automatically fall back to the local **SQLite** database (`retinascan_ai.db`) if MySQL is unavailable.

### 5. Launch the Web App

```bash
streamlit run webapp/newapp.py
```

The RetinaScan AI dashboard will open at `http://localhost:8501`.

---

## Project Structure

```
Diabetic-Retinopathy/
│
├── notebooks/                          # End-to-end research pipeline (run in order)
│   ├── 01_eda.ipynb                    # Exploratory Data Analysis on APTOS + EyePACS
│   ├── 02_preprocess.ipynb             # APTOS preprocessing & dataset building
│   ├── 03_baseline_model.ipynb         # Initial CNN baseline
│   ├── 04_Preprocess_eyepacs.ipynb     # EyePACS preprocessing & merge strategy
│   ├── 05_efficientNet_training.ipynb  # Main EfficientNetB4 training notebook
│   ├── 06_gradcam_efficientnetb4.ipynb # Grad-CAM generation & analysis
│   └── 07_database_integration.ipynb   # DB schema, insertions & scan tracking
│
├── webapp/
│   └── newapp.py                       # Streamlit clinical dashboard (main app)
│
├── src/                                # Core Python package
│   ├── preprocess.py                   # Ben Graham preprocessing pipeline
│   ├── model_utils.py                  # EfficientNetB4 builder, Focal Loss, callbacks
│   ├── gradcam_utils.py                # Grad-CAM engine (split-execution pattern)
│   ├── database.py                     # MySQL scan record management
│   ├── new_database.py                 # Extended DB with model versioning & JSON probs
│   ├── data_loader.py                  # Training data generators & augmentation
│   └── migrate_to_sqlite.py            # MySQL → SQLite migration utility
│
├── tests/
│   └── test_validation.py              # Unit tests for retinal image validation
│
├── scripts/
│   └── check_exts.py                   # Dataset extension audit utility
│
├── results.json                        # Final model metrics snapshot
├── retinascan_ai.db                    # SQLite fallback database
├── environment.yml                     # Conda environment spec (Python 3.11)
├── requirements.txt                    # pip dependencies
└── README.md
```

---

## Research Notebooks

The `notebooks/` directory contains the full end-to-end machine learning pipeline, designed to be run **sequentially**:

| # | Notebook | Purpose |
|:---:|---|---|
| 01 | `01_eda.ipynb` | Dataset exploration, class distribution analysis, sample visualization |
| 02 | `02_preprocess.ipynb` | Ben Graham pipeline on APTOS 2019; quality inspection |
| 03 | `03_baseline_model.ipynb` | Simple CNN baseline to establish a performance floor |
| 04 | `04_Preprocess_eyepacs.ipynb` | EyePACS preprocessing, dataset merge, final split |
| 05 | `05_efficientNet_training.ipynb` | **Core training** — all three phases + ensemble |
| 06 | `06_gradcam_efficientnetb4.ipynb` | Grad-CAM heatmap generation, per-class analysis, overlay export |
| 07 | `07_database_integration.ipynb` | Database schema setup, batch scan insertion, analytics |

---

## Technical Deep Dive

### Preprocessing Pipeline (Ben Graham Method)

```
Raw Fundus Image
      │
      ▼
(1) Circular Crop         → Mask non-retinal corners with Gaussian-blurred boundary
      │
      ▼
(2) Black Border Removal  → Crop tight bounding box around non-zero pixels
      │
      ▼
(3) Local Avg Subtraction → addWeighted(img, 4, GaussianBlur(img), -4, 128)
      │                     Normalizes illumination & reveals fine vessel structure
      ▼
(4) Resize → 380×380      → EfficientNetB4's native input resolution
      │
      ▼
(5) CLAHE Enhancement     → Contrast Limited AHE in LAB color space (L channel)
      │                     clip_limit=2.0, tile_grid=(8×8)
      ▼
(6) float32 [0–255]       → NO division by 255 — EfficientNetB4 normalizes internally
      │
      ▼
   model.predict()
```

### Grad-CAM Implementation

The Grad-CAM engine targets `top_activation` — the last spatial activation layer inside EfficientNetB4 before `GlobalAveragePooling2D`. It uses a "split-execution" pattern to handle the nested `keras.Model` architecture:

```python
# Build gradient model exposing conv features
grad_model = keras.Model(
    inputs=base_model.input,
    outputs=[conv_layer.output, base_model.output]
)

# Compute gradients of the predicted class score w.r.t. conv activations
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img, training=False)
    loss = predictions[:, pred_grade]

grads = tape.gradient(loss, conv_outputs)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # Global Average Pooling
heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]  # Weighted sum
```

Post-processing: ReLU → Normalize → Resize to 380×380 → Gaussian Blur → COLORMAP_JET overlay.

### Database Schema

```sql
CREATE TABLE scans (
    id                INT AUTO_INCREMENT PRIMARY KEY,
    patient_name      VARCHAR(100),
    patient_age       INT,
    eye_side          VARCHAR(20),         -- 'Left Eye' | 'Right Eye' | 'Both'
    grade             INT,                 -- 0–4
    grade_name        VARCHAR(50),         -- 'No DR' → 'Proliferative DR'
    confidence        FLOAT,               -- Softmax score for predicted class
    all_probabilities JSON,                -- Full 5-class probability distribution
    gradcam_path      VARCHAR(500),        -- Path to saved Grad-CAM overlay PNG
    model_version     VARCHAR(100),        -- 'EfficientNetB4_v82pct'
    risk_level        VARCHAR(30),         -- 'Low' | 'Medium' | 'High' | 'Critical'
    scan_date         DATETIME,
    notes             TEXT
);
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `tensorflow-cpu` | 2.17.1 | Model training & inference |
| `streamlit` | latest | Web application framework |
| `opencv-python-headless` | latest | Image preprocessing & Grad-CAM overlays |
| `mysql-connector-python` | latest | MySQL database connectivity |
| `numpy` | latest | Numerical operations |
| `matplotlib` | latest | Visualization in notebooks |
| `Pillow` | latest | Image I/O for Streamlit upload |

---

## Datasets

| Dataset | Source | Images | Classes |
|---|---|:---:|:---:|
| **APTOS 2019 Blindness Detection** | [Kaggle](https://www.kaggle.com/c/aptos2019-blindness-detection) | ~3,662 | 5 |
| **Diabetic Retinopathy Detection 2015** | [Kaggle / EyePACS](https://www.kaggle.com/c/diabetic-retinopathy-detection) | ~35,000+ | 5 |

Both datasets use the same 0–4 DR grading scale (International Clinical DR Scale). The two datasets are merged and preprocessed with the Ben Graham pipeline at 380×380 resolution before training.

---

## Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## Author

**Jeet Paghdar**

- GitHub: [@Jeet-Paghdar](https://github.com/Jeet-Paghdar)
- Location: Gujarat, India

---

## Acknowledgements

- **APTOS 2019** organizers and Asia Pacific Tele-Ophthalmology Society
- **EyePACS** / Kaggle 2015 DR Detection Challenge
- **Ben Graham** — for the preprocessing technique used in competition-winning solutions
- **Google Brain** — for the EfficientNet architecture
- **TensorFlow / Keras** team
