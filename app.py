from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load the model
model_path = 'model/autoencoder_model.h5'
autoencoder = tf.keras.models.load_model(model_path)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Read the image
    img = Image.open(io.BytesIO(file.read())).convert('L')
    img = img.resize((128, 128))
    img = np.array(img).astype('float32') / 255.0
    img = np.expand_dims(img, axis=[0, -1])  # Add batch and channel dimensions

    # Predict
    reconstructed_img = autoencoder.predict(img)
    reconstructed_img = (reconstructed_img[0] * 255).astype('uint8')
    reconstructed_img = Image.fromarray(reconstructed_img.squeeze(), mode='L')

    # Convert image to bytes
    img_byte_arr = io.BytesIO()
    reconstructed_img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    return jsonify({'image': img_byte_arr.hex()})

if __name__ == '__main__':
    app.run(debug=True)
