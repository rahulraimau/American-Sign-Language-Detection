import os
from flask import Flask, request, jsonify
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'asl_classifier_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Define categories (must match the order used during training)
categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']
IMG_SIZE = 64

@app.route('/')
def home():
    return "ASL Classifier API. Use /predict to send an image."

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            # Read the image
            in_memory_file = file.read()
            np_array = np.frombuffer(in_memory_file, np.uint8)
            img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

            if img is None:
                return jsonify({'error': 'Could not decode image'}), 400

            # Preprocess the image
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
            img = np.expand_dims(img, axis=0)  # Add batch dimension

            # Make prediction
            predictions = model.predict(img)
            predicted_class_index = np.argmax(predictions)
            predicted_label = categories[predicted_class_index]
            confidence = float(np.max(predictions))

            return jsonify({
                'predicted_label': predicted_label,
                'confidence': confidence
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # For local development: app.run(debug=True)
    # For Render deployment, Render will set PORT env variable
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
