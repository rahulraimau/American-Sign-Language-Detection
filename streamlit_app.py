import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="ASL Sign Classifier", layout="centered")

st.title("🧠 American Sign Language (ASL) Classifier")
st.write("Upload an image or use your webcam to classify ASL signs.")

API_URL = "https://american-sign-language-detection-4ylu.onrender.com/predict"

# File uploader
uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

# Webcam capture
capture = st.camera_input("📷 Or take a picture")

img_data = None
if uploaded_file:
    img_data = uploaded_file.read()
elif capture:
    img_data = capture.getvalue()

if img_data:
    st.image(Image.open(io.BytesIO(img_data)), caption="Preview", use_column_width=True)
    with st.spinner("Classifying..."):
        response = requests.post(API_URL, files={"file": ("image.jpg", img_data, "image/jpeg")})
        if response.status_code == 200:
            prediction = response.json().get("prediction", "Unknown")
            st.success(f"🧾 Predicted Sign: **{prediction}**")
        else:
            st.error("❌ Failed to get prediction from the API.")
