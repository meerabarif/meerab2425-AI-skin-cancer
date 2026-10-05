import os
import json
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
import gdown

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Skin Lesion Classifier",
    page_icon="🔬",
    layout="centered"
)

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
MODEL_PATH = "model.keras"
INFO_PATH = "class_names.json"
FILE_ID = "1H_kwXv6AO-Ran6XoZL-T5zCUfSCbapwi"

# ---------------------------------------------------------
# Load Model + Metadata
# ---------------------------------------------------------
@st.cache_resource
def load_assets():

    # Download model if not exists
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model weights from Google Drive...")
        url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

    # Load model
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # Load metadata
    with open(INFO_PATH, "r") as f:
        meta = json.load(f)

    return model, meta


model, meta = load_assets()

class_names = meta.get("class_names", [])
img_size = tuple(meta.get("img_size", [224, 224]))

# ---------------------------------------------------------
# UI Header
# ---------------------------------------------------------
st.title("🔬 Skin Lesion Classifier")
st.caption(f"Model: **{meta.get('model', 'CNN')}** | Classes: {', '.join(class_names)}")

st.warning(
    "Educational demo only. NOT a medical device. "
    "Consult a dermatologist for diagnosis."
)

# Sidebar
threshold = st.sidebar.slider(
    "Decision threshold (malignant)",
    0.05, 0.95, 0.50, 0.05
)

st.sidebar.write(
    "Lower threshold = more sensitive (catches more malignant cases)"
)

st.sidebar.json(meta.get("test_metrics", {}))

# ---------------------------------------------------------
# Prediction Logic
# ---------------------------------------------------------
uploaded = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])

if uploaded:

    # Show image
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = image.resize(img_size)
    arr = np.asarray(img, dtype=np.float32)[None, ...] / 255.0

    # Prediction
    pred = model.predict(arr, verbose=0)[0][0]

    # Decision
    if pred >= threshold:
        label = class_names[1]
        conf = pred
    else:
        label = class_names[0]
        conf = 1 - pred

    # Output
    st.subheader("Results")
    st.write(f"**Prediction:** `{label.upper()}`")
    st.metric("Confidence Score", f"{conf * 100:.2f}%")
