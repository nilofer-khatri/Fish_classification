import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("fish_best_model.keras")
    return model

model = load_model()

# Class labels (change these to your fish species names)
CLASS_NAMES = ['Black Sea Sprat', 'Gilt-Head Bream', 'Hourse Mackerel', 'Red Mullet', 'Red Sea Bream', 'Sea Bass', 'Shrimp', 'Striped Red Mullet', 'Trout']

st.title("🐟 Fish Species Classifier")
st.markdown("Upload an image of a fish, and the model will predict its species!")

# File uploader
uploaded_file = st.file_uploader("Choose a fish image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = image.resize((224, 224))  # same size used for training
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions)

    st.write(f"**Prediction:** {predicted_class}")
    st.write(f"**Confidence:** {confidence*100:.2f}%")
