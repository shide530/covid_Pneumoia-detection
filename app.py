import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="COVID-19 Detection from Chest X-Rays",
    page_icon="🩺",
    layout="centered"
)

@st.cache_resource
def load_my_model():
    # Load the saved Keras model
    return tf.keras.models.load_model('covid19_model.keras')

try:
    model = load_my_model()
    st.success("Model loaded successfully!")
typeof_error as e:
    st.error(f"Error loading model. Please ensure 'covid19_model.keras' is in the same directory: {e}")

st.title("🩺 Chest X-Ray COVID-19 Classifier")
st.write("Upload a Chest X-ray image to detect the probability of COVID-19 infection.")

uploaded_file = st.file_uploader("Choose a Chest X-Ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the image
    image_data = Image.open(uploaded_file).convert('RGB')
    st.image(image_data, caption='Uploaded Chest X-Ray', use_column_width=True)
    
    # Preprocess image to match training specification
    # Resize to (150, 150) and scale values to [0, 1]
    resized_img = image_data.resize((150, 150))
    img_array = np.array(resized_img) / 255.0
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    
    if st.button("Run Diagnostic Inference"):
        with st.spinner("Analyzing chest scan..."):
            # Predict probability
            probability = model.predict(img_array)[0][0]
            
            # Map predicted class and confidence
            if probability >= 0.5:
                label = "COVID-19 Positive (1)"
                confidence = probability
                st.error(f"Result: {label} with {confidence:.2%} confidence.")
            else:
                label = "Normal / Healthy (0)"
                confidence = 1.0 - probability
                st.success(f"Result: {label} with {confidence:.2%} confidence.")
