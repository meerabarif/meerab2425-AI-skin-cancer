import os
import json
import gdown
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Skin Lesion Classifier",
    page_icon="🔬",
    layout="centered"
)

# ---------------------------------------------------------
# Configuration & File Setup
# ---------------------------------------------------------
MODEL_PATH = "model.keras"
FILE_ID = "1H_kwXv6AO-Ran6XoZL-T5zCUfSCbapwi"
INFO_PATH = "class_names.json"

@st.cache_resource
def load_assets():
    """Downloads model from Google Drive if missing and loads assets."""

    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model weights from Google Drive...")

        url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    with open(INFO_PATH, "r") as f:
        meta = json.load(f)

    return model, meta


# Load model and metadata
model, meta = load_assets()

class_names = meta.get("class_names", [])
img_size = tuple(meta.get("img_size", [224, 224]))
# ---------------------------------------------------------
# UI & Header Layout
# ---------------------------------------------------------
st.title("🔬 Skin Lesion Classifier")
st.caption(f"Model: **{meta.get('model', 'CNN')}** | Classes: ({', '.join(class_names)})")
st.warning("Educational demo only. This is NOT a medical device and must not be used for diagnosis. Consult a dermatologist for any skin concern.")

# Sidebar Settings
threshold = st.sidebar.slider("Decision threshold (malignant)", 0.05, 0.95, 0.50, 0.05)
st.sidebar.write("Lower threshold = more sensitive (catches more malignant cases, more false alarms)")
st.sidebar.json(meta.get("test_metrics", {}))

# ---------------------------------------------------------
# Image Classification Logic
# ---------------------------------------------------------
uploaded = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    
    arr = np.asarray(image.resize(img_size), dtype=np.float32)[None, ...]
    p_pos = float(model.predict(arr, verbose=0)[0][0])
    
    label = class_names[1] if p_pos >= threshold else class_names[0]
    conf = p_pos if label == class_names[1] else 1 - p_pos
    
    st.subheader("Results")
    st.write(f"**Prediction:** `{label.upper()}`")
    st.metric(label="Confidence Score", value=f"{conf * 100:.2f}%")
