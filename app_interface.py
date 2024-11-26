from flask import Flask, request, render_template, jsonify
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = './models/bansos_model_ai.keras'
model = load_model(MODEL_PATH)

UPLOAD_FOLDER = 'uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

    img_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(img_path)
    
    img_array = prepare_image(img_path)
    prediction = model.predict(img_array)
    # threshold = 0.3 
    # prediction = (prediction > threshold).astype(int)

    print(f'Prediction[0]: {prediction[0]}')

    eligibility = 'Eligible' if prediction[0] < 0.3 else 'Not Eligible'

    return jsonify({'prediction': eligibility, 'confidence': float(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
