import streamlit as st
import requests

st.title("ASL Sign Language Classifier")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Predicting...")

    # Send to backend
    response = requests.post("http://localhost:5000/predict", files={"file": uploaded_file})
    if response.status_code == 200:
        st.success(f"Prediction: {response.json().get('class')}")
    else:
        st.error("Failed to get prediction from backend.")
