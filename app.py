# app.py
# ------------------------------
# Streamlit UI for Eye_model.h5
# Shows: prediction, probabilities, and training parameters
# ------------------------------

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import time
import matplotlib.pyplot as plt

# -------------- CONFIG --------------
MODEL_PATH = "Eye_model.h5"   # <-- Ensure this file is uploaded in your repo
IMG_SIZE   = 224
CLASS_NAMES = ["healthy", "conjunctivitis", "jaundice"]

MODEL_PARAMETERS = {
    "Base model": "MobileNetV2 (frozen) + GAP + Dense(128) + Dense(3, softmax)",
    "Input size": f"{IMG_SIZE} x {IMG_SIZE}",
    "Pre-processing": "Rescale 1/255",
    "Augmentation": "flip, zoom, rotation, shear (during training)",
    "Optimizer": "Adam",
    "Loss": "Categorical Crossentropy",
    "Metrics": "Accuracy",
    "Class weights": "Yes (to balance jaundice)",
    "Epochs": "≈ 15–20",
    "Batch size": "16 (adjust yours if different)"
}
# ------------------------------------

CUSTOM_CSS = """
<style>
    .main > div {
        padding-top: 0rem;
        padding-bottom: 0rem;
    }
    .title {
        text-align: center;
        font-size: 2rem;
        font-weight: 800;
        color: #0F4C81;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        font-size: 1rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .pred-box {
        background: #ecf5ff;
        border-left: 6px solid #0F4C81;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin: 0.8rem 0;
        font-size: 1.05rem;
    }
    .prob-item {
        margin: 0.15rem 0;
        font-size: 0.95rem;
    }
    .footer-note {
        color: #aaa; 
        font-size: 0.8rem; 
        text-align:center; 
        margin-top: 2rem;
    }
</style>
"""

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def preprocess(img: Image.Image, img_size: int = IMG_SIZE):
    img = img.convert("RGB").resize((img_size, img_size))
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

def plot_bar(probs, labels):
    fig, ax = plt.subplots()
    ax.bar(range(len(probs)), probs)
    ax.set_xticks(range(len(probs)))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim([0, 1])
    for i, p in enumerate(probs):
        ax.text(i, p + 0.01, f"{p:.2f}", ha='center', va='bottom', fontsize=9)
    ax.set_ylabel("Probability")
    st.pyplot(fig)

# ---------- UI ----------
st.set_page_config(page_title="Eye Disease Classifier", page_icon="👁️", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown('<div class="title">👁️ Eye Disease Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Healthy • Conjunctivitis • Jaundice</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("📦 Model / Training Parameters")
    for k, v in MODEL_PARAMETERS.items():
        st.markdown(f"**{k}**: {v}")
    st.markdown("---")
    st.subheader("Class index → label")
    for i, c in enumerate(CLASS_NAMES):
        st.write(f"{i} → {c}")

with st.spinner("Loading model..."):
    model = load_model()

uploaded = st.file_uploader("📤 Upload an eye image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    pil_img = Image.open(uploaded)
    col_img, col_out = st.columns([1, 1])

    with col_img:
        st.image(pil_img, caption="Uploaded image", use_column_width=True)

    with col_out:
        x = preprocess(pil_img)
        start = time.time()
        probs = model.predict(x)[0]
        infer_ms = (time.time() - start) * 1000

        pred_idx = int(np.argmax(probs))
        pred_label = CLASS_NAMES[pred_idx]

        st.markdown(f'<div class="pred-box">✅ <b>Prediction:</b> {pred_label}<br/>🕒 Inference time: {infer_ms:.2f} ms</div>', unsafe_allow_html=True)

        st.subheader("🔢 Class Probabilities")
        for cls, p in zip(CLASS_NAMES, probs):
            st.markdown(f'<div class="prob-item">• <b>{cls}:</b> {p:.4f}</div>', unsafe_allow_html=True)

        st.subheader("📊 Probability Chart")
        plot_bar(probs, CLASS_NAMES)

st.markdown('<div class="footer-note">Built with Streamlit • TensorFlow</div>', unsafe_allow_html=True)
