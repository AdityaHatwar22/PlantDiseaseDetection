import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("model/plant_model.h5")

# Class labels
classes = ["Healthy", "Early Blight", "Late Blight"]

st.title("🌿 Plant Disease Detection")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((224,224))
    img = np.array(img) / 255.0
    img = img.reshape(1,224,224,3)

    # Prediction
    pred = model.predict(img)
    class_idx = np.argmax(pred)

    st.success(f"Prediction: {classes[class_idx]}")
    st.info(f"Confidence: {round(100 * np.max(pred), 2)}%")