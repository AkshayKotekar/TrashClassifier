import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image as PILImage
import os

IMG_SIZE = 224 
class_names = ['organic', 'plastic', 'metal', 'paper', 'glass'] 
saved_model_path = r'C:\Users\aksha\OneDrive\Desktop\Exp\guiHopeful\efficientnetb0_trash_classifiernew.keras' 

@st.cache_resource
def load_model():
    model = None
    if not os.path.exists(saved_model_path):
        st.error(f"Model file not found at '{saved_model_path}'. Please place it in the same directory as this script.")
    else:
        try:
            model = tf.keras.models.load_model(saved_model_path)
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {e}. Please ensure the file is a valid .keras model.")
    return model

model = load_model()

def predict_image_class(image_data, model, class_names, img_size):
    if model is None:
        return "Model not loaded", 0.0
    if image_data is None:
        return "No image provided", 0.0

    try:

        img = PILImage.open(image_data).convert('RGB')

        img = img.resize((img_size, img_size))

        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) 

        predictions = model.predict(img_array, verbose=0)
        predicted_class_index = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        predicted_class_name = class_names[predicted_class_index]

        return predicted_class_name, float(confidence)

    except Exception as e:
        st.error(f"Error processing image for prediction: {e}")
        return "Error", 0.0

st.set_page_config(page_title="Real-time Trash Classifier", layout="centered")
st.title("🗑️ Real-time Trash Classifier")
st.write("Upload an image or use your webcam to classify trash into categories: organic, plastic, metal, paper, or glass.")

st.subheader("Upload an Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    predicted_class, confidence = predict_image_class(uploaded_file, model, class_names, IMG_SIZE)

    if predicted_class != "Error":
        st.success(f"Prediction: **{predicted_class}** with confidence **{confidence:.2f}**")
    else:
        st.warning("Could not make a prediction.")

st.markdown("--- ")

st.subheader("Live Webcam Prediction")

camera_image = st.camera_input("Take a picture")
if camera_image:
    st.image(camera_image)
    predicted_class_webcam, confidence_webcam = predict_image_class(camera_image, model, class_names, IMG_SIZE)
    if predicted_class_webcam != "Error":
        st.success(f"Webcam Prediction: **{predicted_class_webcam}** with confidence **{confidence_webcam:.2f}**")
    else:
        st.warning("Could not make a webcam prediction.")
