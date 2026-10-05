import os
import json
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
import gdown

# ---------------------------------------------------------
# Page Setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="Skin Lesion Classifier",
    page_icon="🔬",
    layout="centered"
)

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
MODEL_PATH = "model.h5"
INFO_PATH = "class_names.json"
FILE_ID = "1H_kwXv6AO-Ran6XoZL-T5zCUfSCbapwi"

# ---------------------------------------------------------
# Load Model + Metadata
# ---------------------------------------------------------
@st.cache_resource
def load_assets():
@st.cache_resource
def load_assets():

    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)  # force fresh download

    st.info("Downloading fresh model...")

    url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    with open(INFO_PATH, "r") as f:
        meta = json.load(f)

    return model, meta

# ---------------------------------------------------------
# UI Header
# ---------------------------------------------------------
st.title("🔬 Skin Lesion Classifier")
st.caption(f"Model: {meta.get('model', 'CNN')} | Classes: {', '.join(class_names)}")

st.warning("Educational demo only. NOT a medical device.")

threshold = st.sidebar.slider("Decision Threshold", 0.05, 0.95, 0.50, 0.05)

st.sidebar.json(meta.get("test_metrics", {}))

# ---------------------------------------------------------
# Upload Image
# ---------------------------------------------------------
uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded:

    # Show image
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = image.resize(img_size)
    arr = np.asarray(img, dtype=np.float32)[None, ...] / 255.0

    # Prediction
    preds = model.predict(arr, verbose=0)[0]

    # Best class
    class_index = np.argmax(preds)
    label = class_names[class_index]
    confidence = float(np.max(preds))

    # Display result
    st.subheader("Result")
    st.write(f"**Prediction:** `{label.upper()}`")
    st.metric("Confidence", f"{confidence * 100:.2f}%")

    # Probabilities
    st.subheader("Class Probabilities")

    prob_dict = {}
    for i in range(len(class_names)):
        prob_dict[class_names[i]] = float(preds[i])
        st.write(f"{class_names[i]}: {preds[i]*100:.2f}%")

    st.bar_chart(prob_dict)
