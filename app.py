import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Skin Lesion Classifier", page_icon="🩺", layout="centered")

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("model.keras")
    with open("class_names.json") as f:
        meta = json.load(f)
    return model, meta

model, meta = load_assets()
class_names = meta["class_names"]          # index 0, index 1 (index 1 = positive / malignant)
img_size = tuple(meta["img_size"])

st.title("🩺 Skin Lesion Classifier")
st.caption(f"Model: **{meta['model']}** | Classes: {', '.join(class_names)}")
st.warning("Educational demo only. This is NOT a medical device and must not be used for diagnosis. "
           "Consult a dermatologist for any skin concern.")

threshold = st.sidebar.slider("Decision threshold (malignant)", 0.05, 0.95, 0.50, 0.05)
st.sidebar.write("Lower threshold = more sensitive (catches more malignant cases, more false alarms).")
st.sidebar.json(meta.get("test_metrics", {}))

uploaded = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    arr = np.asarray(image.resize(img_size), dtype=np.float32)[None, ...]   # raw 0-255, preprocessing is inside the model
    p_pos = float(model.predict(arr, verbose=0)[0][0])
    label = class_names[1] if p_pos >= threshold else class_names[0]
    conf = p_pos if label == class_names[1] else 1 - p_pos

    if label == class_names[1]:
        st.error(f"Prediction: **{label.upper()}**  (confidence {conf:.1%})")
    else:
        st.success(f"Prediction: **{label.upper()}**  (confidence {conf:.1%})")
    st.progress(p_pos, text=f"P({class_names[1]}) = {p_pos:.3f}")
