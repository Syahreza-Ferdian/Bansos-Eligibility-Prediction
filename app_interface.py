from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow import keras
from keras._tf_keras.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

MODEL_PATH = './models/bansos_model_ai.keras'
model = keras.models.load_model(MODEL_PATH)

UPLOAD_FOLDER = 'uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Fungsi untuk memeriksa ekstensi file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang diunggah'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'File tidak ditemukan'}), 400
    
    # Memeriksa apakah ekstensi file valid
    if not allowed_file(file.filename):
        return jsonify({'error': 'Ekstensi file tidak valid. Harap unggah file gambar (PNG, JPG, JPEG)'}), 400
    
    # Menyimpan file dengan nama aman
    file_name = secure_filename(file.filename)
    img_path = os.path.join(UPLOAD_FOLDER, file_name)
    file.save(img_path)
    
    img_array = prepare_image(img_path)
    prediction = model.predict(img_array)

    print(f'Raw Prediction: {prediction[0]}') 

    threshold = 0.8  
    eligibility = 'Eligible' if prediction[0] >= threshold else 'Not Eligible'

    return jsonify({'prediction': eligibility, 'confidence': float(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)