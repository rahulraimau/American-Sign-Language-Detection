import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('asl_classifier_model.h5')
    return model

model = load_model()

# Define categories (must match the order used during training)
categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']
IMG_SIZE = 64

st.title("ASL Sign Language Classifier")
st.write("Upload an image of an ASL sign and the model will predict the letter.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image for the model
    opencv_image = np.array(image)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR) # Convert PIL RGB to OpenCV BGR

    img_resized = cv2.resize(opencv_image, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0) # Add batch dimension

    # Make prediction
    predictions = model.predict(img_input)
    predicted_class_index = np.argmax(predictions)
    predicted_label = categories[predicted_class_index]
    confidence = np.max(predictions)

    st.success(f"Prediction: **{predicted_label}** with {confidence*100:.2f}% confidence.")

    st.subheader("All Class Probabilities:")
    prob_df = pd.DataFrame({
        'Class': categories,
        'Probability': predictions[0]
    }).sort_values(by='Probability', ascending=False).reset_index(drop=True)
    st.dataframe(prob_df)
