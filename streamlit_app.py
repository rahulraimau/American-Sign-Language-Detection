import streamlit as st
import requests
from PIL import Image
import io

# Replace this with your actual deployed Flask API URL
API_URL = "https://your-app-name.onrender.com/predict"

st.set_page_config(page_title="ASL Sign Classifier", layout="centered")
st.title("🤟 ASL Alphabet Sign Classifier")
st.write("Upload an image of a hand showing a sign from the ASL alphabet.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict"):
        st.write("Processing...")

        # Convert image to byte stream and send POST request
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        files = {"image": img_bytes}
        try:
            response = requests.post(API_URL, files=files)
            if response.status_code == 200:
                result = response.json()
                st.success(f"✅ Predicted Sign: **{result['prediction']}**")
            else:
                st.error(f"❌ Error: {response.json().get('error', 'Unknown error')}")
        except Exception as e:
            st.error(f"🚨 Request Failed: {e}")
