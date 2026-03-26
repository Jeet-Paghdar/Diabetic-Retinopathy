"""
RetinaScan AI — Streamlit Web Application
==========================================
Diabetic Retinopathy Detection using Ensemble Deep Learning
Run: streamlit run webapp/app.py
""" 

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'          # Suppress TF info/warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'          # Suppress oneDNN notice

import streamlit as st
import numpy as np
import cv2
import sys
import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow import keras
from PIL import Image
import io

# ── Path Setup ────────────────────────────────────────────────────────────────
WEBAPP_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(WEBAPP_DIR)  # Go up from webapp/ to project root
sys.path.insert(0, os.path.join(PROJECT_DIR, 'src'))

from database import (
    setup_database, insert_scan, get_all_scans,
    get_stats, get_scans_by_name, delete_scan, get_scan_by_id
)
from preprocess import is_retinal_image

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RetinaScan AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS — Warm Medical Theme ──────────────────────────────────────────
st.markdown("""
<style>
    /* ── Import Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Main Background ── */
    .stApp {
        background: linear-gradient(135deg, #FFF5F0 0%, #FFF0E6 50%, #FFEEE0 100%);
    }

    /* ── Force Dark Text in Main Content ── */
    .stApp .stMarkdown h1, .stApp .stMarkdown h2,
    .stApp .stMarkdown h3, .stApp .stMarkdown h4,
    .stApp .stMarkdown h5, .stApp .stMarkdown h6 {
        color: #2C3E50 !important;
    }
    .stApp .stMarkdown p,
    .stApp .stMarkdown li,
    .stApp .stMarkdown span {
        color: #333333 !important;
    }
    .stApp label, .stApp .stTextInput label,
    .stApp .stNumberInput label,
    .stApp .stSelectbox label,
    .stApp .stFileUploader label {
        color: #2C3E50 !important;
    }
    .stApp .stTextInput input,
    .stApp .stNumberInput input,
    .stApp .stSelectbox [data-baseweb="select"] span {
        color: #FFFFFF !important;
    }
    .stApp hr {
        border-color: #E0D0C0 !important;
    }
    .stApp .stSpinner > div {
        color: #2C3E50 !important;
    }
    .result-card, .result-card h2, .result-card h3,
    .result-card td, .result-card p, .result-card span,
    .result-card table {
        color: #2C3E50 !important;
    }
    .result-card td[style*="color: #7F8C8D"] {
        color: #7F8C8D !important;
    }
    .advice-box, .advice-box strong, .advice-box br ~ * {
        color: #333333 !important;
    }
    .stat-card, .stat-card h4, .stat-card li,
    .stat-card div {
        color: #2C3E50 !important;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #C0392B 0%, #E74C3C 40%, #F39C12 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stRadio label span {
        color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
        background: rgba(255,255,255,0.15);
        border-radius: 8px;
        padding: 8px 16px;
        margin: 4px 0;
        transition: all 0.3s ease;
    }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
        background: rgba(255,255,255,0.3);
    }

    /* ── Stat Cards ── */
    .stat-card {
        background: white;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(192, 57, 43, 0.08);
        border-left: 5px solid;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stat-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(192, 57, 43, 0.15);
    }
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 8px 0;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #7F8C8D;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ── Grade Badges ── */
    .grade-badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1rem;
        letter-spacing: 0.5px;
    }
    .grade-0 { background: #E8F8F5; color: #1ABC9C; border: 2px solid #1ABC9C; }
    .grade-1 { background: #FEF9E7; color: #F39C12; border: 2px solid #F39C12; }
    .grade-2 { background: #FDF2E9; color: #E67E22; border: 2px solid #E67E22; }
    .grade-3 { background: #FDEDEC; color: #E74C3C; border: 2px solid #E74C3C; }
    .grade-4 { background: #F9EBEA; color: #C0392B; border: 2px solid #C0392B; }

    /* ── Section Headers ── */
    .section-header {
        background: white;
        border-radius: 12px;
        padding: 20px 28px;
        margin-bottom: 24px;
        border-left: 5px solid #E74C3C;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    }
    .section-header h2 {
        color: #C0392B;
        margin: 0;
        font-weight: 600;
    }
    .section-header p {
        color: #7F8C8D;
        margin: 4px 0 0 0;
    }

    /* ── Result Card ── */
    .result-card {
        background: white;
        border-radius: 16px;
        padding: 28px;
        box-shadow: 0 4px 24px rgba(192, 57, 43, 0.1);
        border-top: 4px solid #E74C3C;
    }

    /* ── Advice Box ── */
    .advice-box {
        background: linear-gradient(135deg, #FFF5F0, #FFEEE0);
        border-radius: 12px;
        padding: 20px;
        border-left: 4px solid #F39C12;
        margin-top: 16px;
    }

    /* ── Validation Badges ── */
    .valid-badge {
        background: #E8F8F5;
        color: #1ABC9C;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 600;
        display: inline-block;
    }
    .invalid-badge {
        background: #FDEDEC;
        color: #E74C3C;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 600;
        display: inline-block;
    }

    /* ── Hide Streamlit Defaults ── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    /* ── Button Styling ── */
    .stButton > button {
        background: linear-gradient(135deg, #E74C3C, #C0392B);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 28px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #C0392B, #A93226);
        box-shadow: 0 4px 15px rgba(231, 76, 60, 0.4);
        transform: translateY(-1px);
    }

    /* ── File Uploader ── */
    .stFileUploader {
        border: 2px dashed #E74C3C;
        border-radius: 12px;
        padding: 10px;
    }

    /* ── Disable Streamlit Fullscreen Image Button ── */
    button[title="View fullscreen"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────
GRADE_NAMES = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']
GRADE_COLORS = ['#1ABC9C', '#F39C12', '#E67E22', '#E74C3C', '#C0392B']
GRADE_ADVICE = [
    'No diabetic retinopathy detected. Annual screening recommended.',
    'Mild DR detected. Follow-up in 12 months with your ophthalmologist.',
    'Moderate DR detected. Follow-up in 6 months. Consider treatment options.',
    'Severe DR detected. Urgent ophthalmology referral recommended.',
    'Proliferative DR detected. Immediate ophthalmology referral required.'
]
IMG_SIZE = 224
EFFICIENTNET_PATH = os.path.join(PROJECT_DIR, 'models', 'EfficienetImages', 'efficientnet_best.keras')
RESNET_PATH = os.path.join(PROJECT_DIR, 'models', 'ResnetImages', 'resnet_best.keras')


# ── Model Loading (Cached) ───────────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load both models once and cache them across all sessions."""
    eff_model = None
    res_model = None
    errors = []

    if os.path.exists(EFFICIENTNET_PATH):
        eff_model = keras.models.load_model(EFFICIENTNET_PATH)
    else:
        errors.append(f"EfficientNet not found at {EFFICIENTNET_PATH}")

    if os.path.exists(RESNET_PATH):
        res_model = keras.models.load_model(RESNET_PATH)
    else:
        errors.append(f"ResNet50 not found at {RESNET_PATH}")

    return eff_model, res_model, errors


#Helper Functions
def ensemble_predict(img_array, eff_model, res_model):
    """Run ensemble prediction using both models."""
    eff_probs = eff_model.predict(img_array, verbose=0)[0]
    res_probs = res_model.predict(img_array, verbose=0)[0]
    probs = (eff_probs + res_probs) / 2.0
    pred_grade = int(np.argmax(probs))
    confidence = float(probs[pred_grade])
    return probs, pred_grade, confidence, eff_probs, res_probs


def get_gradcam(model, img_array, pred_grade):
    """Generate Grad-CAM heatmap for visualization."""
    try:
        base = model.get_layer('efficientnetb3')
        conv_layer = base.get_layer('top_conv')

        flat_model = keras.models.Model(
            inputs=base.input,
            outputs=[conv_layer.output, base.get_layer('top_activation').output]
        )

        img_tensor = tf.cast(img_array, tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            conv_outputs, activated = flat_model(img_tensor, training=False)
            tape.watch(conv_outputs)

            x = activated
            for layer in model.layers:
                if layer.name == 'efficientnetb3':
                    continue
                try:
                    x = layer(x, training=False)
                except:
                    continue

            predictions = x
            class_score = predictions[:, pred_grade]

        grads = tape.gradient(class_score, conv_outputs)

        if grads is None:
            with tf.GradientTape() as tape2:
                tape2.watch(img_tensor)
                preds = model(img_tensor, training=False)
                class_score = preds[:, pred_grade]
            grads = tape2.gradient(class_score, img_tensor)
            grads = tf.abs(grads)
            heatmap = tf.reduce_max(grads, axis=-1)[0]
            heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
            return heatmap.numpy()

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()

    except Exception as e:
        st.warning(f"Grad-CAM could not be generated: {e}")
        return None


def overlay_heatmap(img, heatmap, alpha=0.4):
    """Overlay Grad-CAM heatmap on image."""
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    img_uint8 = np.uint8(255 * img) if img.max() <= 1.0 else np.uint8(img)
    return cv2.addWeighted(img_uint8, 1 - alpha, heatmap_colored, alpha, 0)


def interpret_gradcam(pred_grade, confidence):
    """Return a patient-friendly explanation for the predicted DR grade."""
    grade_findings = {
        0: {
            'title': 'No Diabetic Retinopathy',
            'detail': ('The model did not detect significant retinal abnormalities. '
                       'The retinal vasculature and overall structure appear within normal limits.'),
            'features': 'No notable micro-aneurysms, haemorrhages, or exudates were identified.',
        },
        1: {
            'title': 'Mild Non-Proliferative DR',
            'detail': ('The model detected early-stage changes suggesting mild '
                       'non-proliferative diabetic retinopathy (NPDR).'),
            'features': 'Possible presence of a small number of micro-aneurysms — '
                        'tiny bulges in retinal blood vessels that can leak fluid.',
        },
        2: {
            'title': 'Moderate Non-Proliferative DR',
            'detail': ('The model identified moderate-stage changes, indicating '
                       'progression of non-proliferative diabetic retinopathy.'),
            'features': 'Possible micro-aneurysms along with dot-blot haemorrhages '
                        'and/or hard exudates (lipid deposits), suggesting increased '
                        'vascular leakage.',
        },
        3: {
            'title': 'Severe Non-Proliferative DR',
            'detail': ('The model found widespread retinal changes consistent with '
                       'severe non-proliferative diabetic retinopathy.'),
            'features': 'Extensive haemorrhages, venous beading, and/or intra-retinal '
                        'microvascular abnormalities (IRMA) across multiple quadrants, '
                        'indicating significant ischemia.',
        },
        4: {
            'title': 'Proliferative Diabetic Retinopathy',
            'detail': ('The model detected advanced proliferative changes — the most '
                       'serious stage of diabetic retinopathy.'),
            'features': 'Signs consistent with neovascularisation (abnormal new blood '
                        'vessel growth), possible vitreous/pre-retinal haemorrhage, '
                        'and/or fibrous tissue proliferation.',
        },
    }

    info = grade_findings.get(pred_grade, grade_findings[0])

    lines = []
    lines.append(f"**Diagnosis: {info['title']}** (Confidence: {confidence*100:.1f}%)")
    lines.append('')
    lines.append(f"{info['detail']}")
    lines.append('')
    lines.append(f"**Key Findings:** {info['features']}")

    return '\n'.join(lines)


# ── Database Init ─────────────────────────────────────────────────────────────
try:
    setup_database()
    db_available = True
except Exception:
    db_available = False



# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 👁️ RetinaScan AI")
    st.markdown("*Diabetic Retinopathy Detection*")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["Dashboard", "Scan & Predict", "Patient Records", "About"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown(
        "<p style='font-size: 0.75rem; opacity: 0.7;'>"
        "EfficientNetB3 + ResNet50<br>Ensemble Model v1.0</p>",
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":
    st.markdown("""
    <div class='section-header'>
        <h2>👁️ RetinaScan AI — Dashboard</h2>
        <p>Overview of all diagnostic scans and statistics</p>
    </div>
    """, unsafe_allow_html=True)

    if db_available:
        stats = get_stats()
        total = stats.get('total', 0)
        dr_detected = stats.get('dr_detected', 0)
        severe = stats.get('severe', 0)
        avg_conf = stats.get('avg_conf', 0)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class='stat-card' style='border-color: #3498DB;'>
                <div class='stat-label'>Total Scans</div>
                <div class='stat-number' style='color: #3498DB;'>{total}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class='stat-card' style='border-color: #E67E22;'>
                <div class='stat-label'>DR Detected</div>
                <div class='stat-number' style='color: #E67E22;'>{dr_detected}</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class='stat-card' style='border-color: #E74C3C;'>
                <div class='stat-label'>Severe Cases</div>
                <div class='stat-number' style='color: #E74C3C;'>{severe}</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class='stat-card' style='border-color: #1ABC9C;'>
                <div class='stat-label'>Avg Confidence</div>
                <div class='stat-number' style='color: #1ABC9C;'>{avg_conf*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Grade distribution chart
        if total > 0:
            records = get_all_scans()
            grade_counts = {i: 0 for i in range(5)}
            for row in records:
                grade = row[4]  # grade column
                if grade in grade_counts:
                    grade_counts[grade] += 1

            st.markdown("""
            <div class='result-card'>
                <h3 style='color: #C0392B; margin-bottom: 16px;'>Grade Distribution</h3>
            </div>
            """, unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(10, 4))
            bars = ax.bar(
                GRADE_NAMES,
                [grade_counts[i] for i in range(5)],
                color=GRADE_COLORS,
                edgecolor='white',
                linewidth=2
            )
            ax.set_ylabel('Number of Patients', fontweight='bold', color='#555')
            ax.set_facecolor('#FFF5F0')
            fig.set_facecolor('#FFF5F0')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#DDD')
            ax.spines['bottom'].set_color('#DDD')
            ax.tick_params(colors='#555')
            for bar, count in zip(bars, [grade_counts[i] for i in range(5)]):
                if count > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                            str(count), ha='center', fontweight='bold', color='#555')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # Recent scans table
        if total > 0:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div class='section-header'>
                <h3 style='color: #C0392B;'>🕐 Recent Scans</h3>
            </div>
            """, unsafe_allow_html=True)

            records = get_all_scans()[:5]
            for row in records:
                name       = row[1] if row[1] else "Unknown"
                age        = row[2] if row[2] is not None else "—"
                eye        = row[3] if row[3] else "—"
                grade      = row[4] if row[4] is not None else 0
                grade_name = row[5] if row[5] else GRADE_NAMES[grade] if isinstance(grade, int) and 0 <= grade <= 4 else "Unknown"
                conf       = row[6] if row[6] is not None else 0.0
                scan_date  = row[7]

                st.markdown(f"""
                <div style='background: white; border-radius: 10px; padding: 14px 20px; margin-bottom: 8px;
                            box-shadow: 0 2px 8px rgba(0,0,0,0.04); border-left: 4px solid {GRADE_COLORS[grade] if isinstance(grade, int) and 0 <= grade <= 4 else "#999"};
                            display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;'>
                    <div style='min-width: 200px;'>
                        <span style='font-weight: 600; font-size: 1.05rem; color: #2C3E50;'>{name}</span>
                        <span style='color: #7F8C8D; font-size: 0.85rem; margin-left: 8px;'>Age {age} · {eye}</span>
                    </div>
                    <div style='display: flex; align-items: center; gap: 24px; flex-wrap: wrap;'>
                        <span class='grade-badge grade-{grade}'>{grade_name}</span>
                        <span style='color: #555; font-weight: 500;'>{conf*100:.1f}%</span>
                        <span style='color: #7F8C8D; font-size: 0.85rem;'>{scan_date.strftime('%d %b %Y') if scan_date else '—'}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Database not connected. Start MySQL and refresh the page.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: SCAN & PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Scan & Predict":
    st.markdown("""
    <div class='section-header'>
        <h2>🔬 Scan & Predict</h2>
        <p>Upload a retinal fundus image to detect diabetic retinopathy</p>
    </div>
    """, unsafe_allow_html=True)

    # Load models
    eff_model, res_model, model_errors = load_models()
    if model_errors:
        for err in model_errors:
            st.error(f"❌ {err}")
        st.stop()

    # ── Patient Details ───────────────────────────────────
    st.markdown("### 👤 Patient Details")
    col1, col2, col3 = st.columns(3)
    with col1:
        patient_name = st.text_input("Patient Name", placeholder="e.g. Ramesh Kumar")
    with col2:
        patient_age = st.number_input("Age", min_value=1, max_value=120, value=45)
    with col3:
        eye_side = st.selectbox("Eye Side", ["Left Eye", "Right Eye", "Both"])
    notes = st.text_input("Clinical Notes (optional)", placeholder="Any relevant notes...")

    st.markdown("---")

    # ── Image Upload ──────────────────────────────────────
    st.markdown("### 📸 Upload Retinal Image")
    uploaded_file = st.file_uploader(
        "Choose a retinal fundus image",
        type=['png', 'jpg', 'jpeg', 'bmp', 'webp', 'WEBP'], 
        help="Upload a retinal fundus photograph for DR grading"
    )

    if uploaded_file is not None:
        # Read the image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is None:
            st.error("❌ Could not read the uploaded image. Please try another file.")
            st.stop()

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # ── Retinal Validation Gate ──
        is_valid, val_conf, val_reason = is_retinal_image(img_bgr)

        col_img, col_info = st.columns([1, 1])
        with col_img:
            st.image(img_rgb, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)

        with col_info:
            if is_valid:
                st.markdown(
                    f"<div class='valid-badge'>✅ Retinal image verified — {val_conf:.0%} confidence</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='invalid-badge'>⚠️ Validation Failed</div>",
                    unsafe_allow_html=True
                )
                st.error(val_reason)
                st.warning("This does not appear to be a retinal fundus image. "
                           "Please upload a valid retinal scan to proceed.")
                st.stop()

        st.markdown("---")

        # ── Predict Button ────────────────────────────────
        if st.button("🔍 Run DR Analysis", use_container_width=True):
            if not patient_name.strip():
                st.warning("Please enter the patient name before running analysis.")
                st.stop()

            with st.spinner("Running ensemble analysis..."):
                # Preprocess
                img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
                img_norm = img_resized.astype(np.float32) / 255.0
                img_array = np.expand_dims(img_norm, axis=0)

                # Predict
                probs, pred_grade, confidence, eff_probs, res_probs = ensemble_predict(
                    img_array, eff_model, res_model
                )

                # Generate Grad-CAM
                heatmap = get_gradcam(eff_model, img_array, pred_grade)

            # Store results in session state so they survive reruns
            st.session_state['prediction'] = {
                'probs': probs,
                'pred_grade': pred_grade,
                'confidence': confidence,
                'eff_probs': eff_probs,
                'res_probs': res_probs,
                'heatmap': heatmap,
                'img_resized': img_resized,
                'img_norm': img_norm,
                'patient_name': patient_name.strip(),
                'patient_age': patient_age,
                'eye_side': eye_side,
                'notes': notes,
            }

        # ── Display Results (from session state) ─────────────
        if 'prediction' in st.session_state:
            p = st.session_state['prediction']
            pred_grade = p['pred_grade']
            confidence = p['confidence']
            probs      = p['probs']
            eff_probs  = p['eff_probs']
            res_probs  = p['res_probs']
            heatmap    = p['heatmap']
            img_resized = p['img_resized']
            img_norm   = p['img_norm']
            saved_name = p['patient_name']
            saved_age  = p['patient_age']
            saved_eye  = p['eye_side']
            saved_notes = p['notes']

            st.markdown(f"""
            <div class='result-card'>
                <h2 style='color: #C0392B; margin-bottom: 20px;'>🩺 Diagnosis Result</h2>
                <table style='width: 100%; font-size: 1.05rem;'>
                    <tr><td style='padding: 6px 0; color: #7F8C8D;'>Patient</td>
                        <td style='font-weight: 600;'>{saved_name}</td></tr>
                    <tr><td style='padding: 6px 0; color: #7F8C8D;'>Age / Eye</td>
                        <td>{saved_age} years / {saved_eye}</td></tr>
                    <tr><td style='padding: 6px 0; color: #7F8C8D;'>Diagnosis</td>
                        <td><span class='grade-badge grade-{pred_grade}'>{GRADE_NAMES[pred_grade]}</span></td></tr>
                    <tr><td style='padding: 6px 0; color: #7F8C8D;'>Confidence</td>
                        <td style='font-weight: 600;'>{confidence*100:.1f}%</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

            # Advice box
            st.markdown(f"""
            <div class='advice-box'>
                <strong>💡 Clinical Advice:</strong><br>{GRADE_ADVICE[pred_grade]}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Model Comparison ──────────────────────────
            col_eff, col_res, col_ens = st.columns(3)
            with col_eff:
                eff_grade = int(np.argmax(eff_probs))
                st.markdown(f"""
                <div class='stat-card' style='border-color: #3498DB;'>
                    <div class='stat-label'>EfficientNetB3</div>
                    <div class='grade-badge grade-{eff_grade}' style='margin: 8px 0;'>{GRADE_NAMES[eff_grade]}</div>
                    <div style='color: #555;'>{eff_probs.max()*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with col_res:
                res_grade = int(np.argmax(res_probs))
                st.markdown(f"""
                <div class='stat-card' style='border-color: #E67E22;'>
                    <div class='stat-label'>ResNet50</div>
                    <div class='grade-badge grade-{res_grade}' style='margin: 8px 0;'>{GRADE_NAMES[res_grade]}</div>
                    <div style='color: #555;'>{res_probs.max()*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with col_ens:
                st.markdown(f"""
                <div class='stat-card' style='border-color: #C0392B;'>
                    <div class='stat-label'>Ensemble (Final)</div>
                    <div class='grade-badge grade-{pred_grade}' style='margin: 8px 0;'>{GRADE_NAMES[pred_grade]}</div>
                    <div style='color: #555;'>{confidence*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Probability Chart ─────────────────────────
            st.markdown("#### 📊 Grade Probabilities")
            fig, ax = plt.subplots(figsize=(10, 3.5))
            bars = ax.barh(GRADE_NAMES[::-1], probs[::-1] * 100,
                           color=GRADE_COLORS[::-1], edgecolor='white', linewidth=2, height=0.6)
            ax.set_xlabel('Probability (%)', fontweight='bold', color='#555')
            ax.set_xlim(0, 100)
            ax.set_facecolor('#FFF5F0')
            fig.set_facecolor('#FFF5F0')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#DDD')
            ax.spines['bottom'].set_color('#DDD')
            ax.tick_params(colors='#555')
            for bar, prob in zip(bars, probs[::-1] * 100):
                if prob > 3:
                    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                            f'{prob:.1f}%', va='center', fontweight='bold', color='#555', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # ── Grad-CAM Visualization ────────────────────
            if heatmap is not None:
                st.markdown("#### 🔥 Grad-CAM — Where the Model Looked")
                overlay = overlay_heatmap(img_norm, heatmap)

                col_orig, col_heat, col_over = st.columns(3)
                with col_orig:
                    st.image(img_resized, caption="Original", use_container_width=True)
                with col_heat:
                    heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
                    heatmap_colored = np.uint8(255 * heatmap_resized)
                    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
                    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                    st.image(heatmap_colored, caption="Heatmap", use_container_width=True)
                with col_over:
                    st.image(overlay, caption="Overlay", use_container_width=True)

                # ── Grad-CAM Interpretation ───────────────
                st.markdown("<br>", unsafe_allow_html=True)
                explanation = interpret_gradcam(pred_grade, confidence)
                st.markdown("""
                <div class='result-card' style='border-top: 4px solid #F39C12;'>
                    <h3 style='color: #C0392B; margin-bottom: 16px;'>Patient Report — Why This Grade?</h3>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(explanation)

            # ── Save to Database ──────────────────────────
            st.markdown("---")
            if db_available:
                if st.button("💾 Save to Patient Database", use_container_width=True):
                    auto_notes = saved_notes if saved_notes.strip() else ""
                    record_id = insert_scan(
                        patient_name=saved_name,
                        patient_age=saved_age,
                        eye_side=saved_eye,
                        grade=pred_grade,
                        confidence=confidence,
                        notes=auto_notes
                    )
                    if record_id:
                        st.success(f"✅ Record saved! ID: {record_id} | {saved_name} | {GRADE_NAMES[pred_grade]}")
                        # Clear prediction from session state after saving
                        del st.session_state['prediction']
                    else:
                        st.error("Failed to save. Check MySQL connection.")
            else:
                st.info("Database not connected. Records cannot be saved.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: PATIENT RECORDS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Patient Records":
    st.markdown("""
    <div class='section-header'>
        <h2>📋 Patient Records</h2>
        <p>View, search, and manage patient scan history</p>
    </div>
    """, unsafe_allow_html=True)

    if not db_available:
        st.warning("⚠️ Database not connected. Start MySQL and refresh.")
        st.stop()

    # Search
    search_name = st.text_input("🔍 Search by patient name", placeholder="Type a name...")

    if search_name.strip():
        records = get_scans_by_name(search_name.strip())
        st.info(f"Found {len(records)} result(s) for '{search_name}'")
    else:
        records = get_all_scans()

    if not records:
        st.info("No records found.")
    else:
        st.markdown(f"**Showing {len(records)} record(s)**")
        st.markdown("<br>", unsafe_allow_html=True)

        total_records = len(records)
        for idx, row in enumerate(records):
            rid        = row[0]  # actual DB id (used for delete)
            display_no = total_records - idx  # newest = highest number
            name       = row[1] if row[1] else "Unknown"
            age        = row[2] if row[2] is not None else "—"
            eye        = row[3] if row[3] else "—"
            grade      = row[4] if row[4] is not None else 0
            grade_name = row[5] if row[5] else GRADE_NAMES[grade] if isinstance(grade, int) and 0 <= grade <= 4 else "Unknown"
            conf       = row[6] if row[6] is not None else 0.0
            scan_date  = row[7]
            notes_text = row[8] if row[8] else ""

            # Format date safely
            date_str = scan_date.strftime('%d %b %Y, %I:%M %p') if scan_date else "—"

            # Format notes for display
            notes_display = notes_text if notes_text.strip() else "No notes"

            st.markdown(f"""
            <div class='result-card' style='margin-bottom: 16px; border-left: 5px solid {GRADE_COLORS[grade] if isinstance(grade, int) and 0 <= grade <= 4 else "#999"}; border-top: none;'>
                <div style='display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;'>
                    <div style='flex: 1; min-width: 200px;'>
                        <p style='color: #7F8C8D; font-size: 0.8rem; margin: 0;'>Record #{display_no}</p>
                        <h3 style='color: #2C3E50; margin: 4px 0 8px 0; font-size: 1.3rem;'>👤 {name}</h3>
                        <table style='font-size: 0.95rem; border-collapse: collapse;'>
                            <tr>
                                <td style='padding: 4px 16px 4px 0; color: #7F8C8D; font-weight: 500;'>Age</td>
                                <td style='padding: 4px 0; color: #2C3E50;'>{age} years</td>
                            </tr>
                            <tr>
                                <td style='padding: 4px 16px 4px 0; color: #7F8C8D; font-weight: 500;'>Eye</td>
                                <td style='padding: 4px 0; color: #2C3E50;'>{eye}</td>
                            </tr>
                            <tr>
                                <td style='padding: 4px 16px 4px 0; color: #7F8C8D; font-weight: 500;'>Scan Date</td>
                                <td style='padding: 4px 0; color: #2C3E50;'>{date_str}</td>
                            </tr>
                        </table>
                    </div>
                    <div style='text-align: center; min-width: 180px; padding: 8px;'>
                        <p style='color: #7F8C8D; font-size: 0.8rem; margin: 0 0 6px 0;'>Diagnosis</p>
                        <span class='grade-badge grade-{grade}'>{grade_name}</span>
                        <p style='font-size: 1.1rem; font-weight: 600; color: #2C3E50; margin: 10px 0 2px 0;'>{conf*100:.1f}%</p>
                        <p style='color: #7F8C8D; font-size: 0.8rem; margin: 0;'>Confidence</p>
                    </div>
                </div>
                <div style='margin-top: 12px; padding-top: 10px; border-top: 1px solid #F0E0D6;'>
                    <p style='color: #7F8C8D; font-size: 0.8rem; margin: 0 0 4px 0;'>📝 Clinical Notes</p>
                    <p style='color: #555; font-size: 0.9rem; margin: 0;'>{notes_display}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Delete button for each record
            if st.button(f"🗑️ Delete Record #{display_no}", key=f"del_{rid}"):
                if delete_scan(rid):
                    st.success(f"Record #{rid} deleted.")
                    st.rerun()
                else:
                    st.error("Delete failed.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "About":
    st.markdown("""
    <div class='section-header'>
        <h2>ℹ️ About RetinaScan AI</h2>
        <p>AI-powered diabetic retinopathy detection system</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='result-card'>
        <h3 style='color: #C0392B;'>👁️ What is Diabetic Retinopathy?</h3>
        <p style='color: #555; line-height: 1.8;'>
            Diabetic Retinopathy (DR) is a diabetes complication that affects the eyes.
            It's caused by damage to the blood vessels of the light-sensitive tissue at the back
            of the eye (retina). It can develop in anyone who has type 1 or type 2 diabetes,
            and is a leading cause of blindness worldwide.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### 🏗️ Model Architecture")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class='stat-card' style='border-color: #3498DB; text-align: left;'>
            <h4 style='color: #3498DB;'>EfficientNetB3</h4>
            <ul style='color: #555;'>
                <li>ImageNet pretrained backbone</li>
                <li>Two-phase training (freeze → fine-tune)</li>
                <li>Global Average Pooling + Dense layers</li>
                <li>Optimized for balanced accuracy</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='stat-card' style='border-color: #E67E22; text-align: left;'>
            <h4 style='color: #E67E22;'>ResNet50</h4>
            <ul style='color: #555;'>
                <li>ImageNet pretrained backbone</li>
                <li>Two-phase training (freeze → fine-tune)</li>
                <li>512 → 256 Dense head with BatchNorm</li>
                <li>Residual connections for deep features</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### 📊 DR Grading Scale")
    for i in range(5):
        st.markdown(
            f"<span class='grade-badge grade-{i}' style='margin: 4px;'>Grade {i}: {GRADE_NAMES[i]}</span>",
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class='advice-box'>
        <strong>⚠️ Disclaimer:</strong><br>
        This tool is designed for screening purposes only and should not replace
        professional medical diagnosis. Always consult a qualified ophthalmologist
        for definitive diagnosis and treatment decisions.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: #999; font-size: 0.85rem;'>
        <p>Built with ❤️ using TensorFlow, Streamlit & MySQL</p>
        <p>Dataset: APTOS 2019 Blindness Detection · Kaggle</p>
    </div>
    """, unsafe_allow_html=True)