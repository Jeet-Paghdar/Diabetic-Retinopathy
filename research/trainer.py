"""
research/trainer.py - Research Benchmarking & Training Orchestrator
===================================================================
Orchestrates the training of literature models (Arora, SVM, ResNet)
on the actual APTOS dataset and compares them against RetinaScan 82%.
"""

import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from architectures import build_arora_b0, build_effnet_svm_extractor, build_revised_resnet50
from preprocessing import apply_sop_preprocessing, apply_standard_normalization, apply_imagenet_scaling

# ── 1. CONFIGURATION ─────────────────────────────────────────────────────────

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = BASE_DIR
DATA_CSV = os.path.join(PROJECT_DIR, 'data', 'raw', 'train.csv')
IMG_DIR  = os.path.join(PROJECT_DIR, 'data', 'raw', 'train_images')
MODEL_PATH = os.path.join(PROJECT_DIR, 'models', 'efficientnet_best.keras')

# Training Params
BATCH_SIZE = 16
EPOCHS = 8  # Balanced for time and accuracy
IMG_SIZE_BASELINE = 380
IMG_SIZE_LIT = 224

# ── 2. DATA LOADER ──────────────────────────────────────────────────────────

def load_research_data(samples_per_class=150):
    """
    Loads a balanced subset of the APTOS dataset for benchmarking.
    """
    if not os.path.exists(DATA_CSV):
        print(f"Error: CSV not found at {DATA_CSV}")
        return None, None, None, None
        
    df = pd.read_csv(DATA_CSV)
    
    # Stratified split for test
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['diagnosis'], random_state=42)
    
    # Balanced Train Subset
    def balance(sub_df, n):
        balanced = []
        for grade in range(5):
            g_df = sub_df[sub_df['diagnosis'] == grade]
            n_samples = min(len(g_df), n)
            balanced.append(g_df.sample(n_samples, random_state=42))
        return pd.concat(balanced).sample(frac=1, random_state=42)
    
    train_balanced = balance(train_df, samples_per_class)
    test_balanced = balance(test_df, 50) # 250 test images
    
    return train_balanced, test_balanced

def get_images_batch(df, preprocessing_fn, target_size=(224, 224)):
    """
    Helper to load and preprocess a batch of images.
    """
    images = []
    labels = []
    for _, row in df.iterrows():
        path = os.path.join(IMG_DIR, f"{row['id_code']}.png")
        if not os.path.exists(path): continue
        
        img = cv2.imread(path)
        if img is None: continue
        
        processed = preprocessing_fn(img, target_size)
        images.append(processed)
        labels.append(row['diagnosis'])
    
    return np.array(images), np.array(labels)

# ── 3. BENCHMARK RUNNER ──────────────────────────────────────────────────────

class ResearchBenchmark:
    def __init__(self):
        self.results = {}
        self.train_df, self.test_df = load_research_data()
        
        print(f"Loading RetinaScan-AI Baseline...")
        self.baseline_model = keras.models.load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

    def run_benchmark(self):
        print("\n" + "="*60)
        print("  STARTING LITERATURE COMPARISON STUDY")
        print("="*60)
        
        # 1. ARORA (2024)
        print("\n[1/4] Training Arora et al. (B0)...")
        model_a = build_arora_b0(input_shape=(224, 224, 3))
        model_a.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        x_train, y_train = get_images_batch(self.train_df, apply_standard_normalization)
        x_test, y_test = get_images_batch(self.test_df, apply_standard_normalization)
        
        model_a.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
        
        preds = np.argmax(model_a.predict(x_test), axis=1)
        self.results['Arora (2024)'] = {
            'Accuracy': accuracy_score(y_test, preds),
            'Kappa': cohen_kappa_score(y_test, preds, weights='quadratic'),
            'Novelty': 'EfficientNetB0 Baseline'
        }

        # 2. EFFNET-SVM (2025)
        print("\n[2/4] Training EffNet-SVM (Hybrid)...")
        extractor = build_effnet_svm_extractor()
        x_train_svm, y_train_svm = get_images_batch(self.train_df, apply_imagenet_scaling)
        x_test_svm, y_test_svm = get_images_batch(self.test_df, apply_imagenet_scaling)
        
        features_train = extractor.predict(x_train_svm)
        features_test = extractor.predict(x_test_svm)
        
        svm = SVC(kernel='rbf', probability=True)
        svm.fit(features_train, y_train_svm)
        
        preds_svm = svm.predict(features_test)
        self.results['EffNet-SVM (2025)'] = {
            'Accuracy': accuracy_score(y_test_svm, preds_svm),
            'Kappa': cohen_kappa_score(y_test_svm, preds_svm, weights='quadratic'),
            'Novelty': 'Hybrid CNN-SVM'
        }

        # 3. LIN & WU (2023) - REVISED RESNET
        print("\n[3/4] Training Revised ResNet-50 (SOP)...")
        model_r = build_revised_resnet50()
        model_r.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        x_train_r, y_train_r = get_images_batch(self.train_df, apply_sop_preprocessing)
        x_test_r, y_test_r = get_images_batch(self.test_df, apply_sop_preprocessing)
        
        model_r.fit(x_train_r, y_train_r, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
        
        preds_r = np.argmax(model_r.predict(x_test_r), axis=1)
        self.results['Lin & Wu (2023)'] = {
            'Accuracy': accuracy_score(y_test_r, preds_r),
            'Kappa': cohen_kappa_score(y_test_r, preds_r, weights='quadratic'),
            'Novelty': 'Feature Merging'
        }

        # 4. RETINASCAN (YOURS)
        print("\n[4/4] Evaluating RetinaScan (82% Baseline)...")
        # Load images at 380x380 for B4
        def raw_load(img, size): return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (380,380)).astype(np.float32)
        x_test_base, y_test_base = get_images_batch(self.test_df, raw_load, target_size=(380,380))
        
        if self.baseline_model:
            preds_base = np.argmax(self.baseline_model.predict(x_test_base), axis=1)
            self.results['RetinaScan (YOURS)'] = {
                'Accuracy': accuracy_score(y_test_base, preds_base),
                'Kappa': cohen_kappa_score(y_test_base, preds_base, weights='quadratic'),
                'Novelty': 'B4 + Focal Loss + XAI'
            }

        self.print_final_report()

    def print_final_report(self):
        print("\n" + "#"*60)
        print("  FINAL RESEARCH VALIDATION LEADERBOARD")
        print("#"*60)
        sorted_res = sorted(self.results.items(), key=lambda x: x[1]['Accuracy'], reverse=True)
        for name, m in sorted_res:
            print(f"{name:<20} | Acc: {m['Accuracy']:.1%} | Kappa: {m['Kappa']:.3f} | {m['Novelty']}")
        print("#"*60)

if __name__ == "__main__":
    runner = ResearchBenchmark()
    runner.run_benchmark()
