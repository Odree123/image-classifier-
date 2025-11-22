
# app.py
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("my_model.h5")

# Class names must match the folders in 'train' (order matters)
class_names = ["fruits", "vegetables"]

st.set_page_config(page_title="Fruit & Vegetable Classifier", layout="centered")
st.title("🥕🍎 Fruit & Vegetable Classifier")
st.write("Enter the name of the image and upload it. The model will predict if it’s a fruit or vegetable.")

# Ask user to type the image name
image_name = st.text_input("Enter the name of the image:")

# Upload image
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption=f"Uploaded Image: {image_name}", width=250)

    # Preprocess image
    img = img.resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 64, 64, 3)

    # Predict
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    confidence = float(np.max(prediction))
    predicted_label = class_names[predicted_index]

    # Show results
    st.write(f"### 🏷 Entered Name: **{image_name}**")
    st.write(f"### 🔍 Predicted Class: **{predicted_label.capitalize()}**")
    st.write(f"### 📊 Confidence: **{confidence*100:.2f}%**")
