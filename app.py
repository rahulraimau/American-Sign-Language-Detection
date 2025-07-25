from flask import Flask, request, jsonify
import numpy as np
import cv2
import tensorflow as tf
import os

app = Flask(__name__)
model = tf.keras.models.load_model("asl_classifier_model.h5")

# Define label names
categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
              'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
              'del', 'nothing', 'space']

IMG_SIZE = 64

@app.route('/')
def home():
    return "ASL Classifier API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    if img is None:
        return jsonify({'error': 'Invalid image'}), 400

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 3)

    predictions = model.predict(img)
    predicted_class = categories[np.argmax(predictions)]

    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
