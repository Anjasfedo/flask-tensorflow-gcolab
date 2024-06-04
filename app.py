import os
from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import io
from model_load import best_model, class_names
# from firebase_admin import credentials
# import firebase_admin 
# from auth_middleware import firebase_authentication_middleware
import datetime
# from firebase_admin import firestore
# from flask_cors import CORS
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# import matplotlib.pyplot as plt

# Set the environment variable to disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)
# CORS(app)

# cred = credentials.Certificate('serviceAccount.json')
# firebase_admin.initialize_app(cred)

# Fungsi untuk menyimpan hasil prediksi ke dalam Firestore
# def save_prediction_to_firestore(predicted_class, confidence):
#     db = firestore.client()
    
#     # Buat dokumen baru di koleksi 'history'
#     new_prediction_ref = db.collection('history').document()
    
#     # Data untuk disimpan
#     data = {
#         'timestamp': datetime.datetime.now(),
#         'predicted_class': predicted_class,
#         'confidence': confidence
#     }
    
#     # Simpan data ke dalam Firestore
#     new_prediction_ref.set(data)

# Fungsi untuk memuat dan memproses gambar
def preprocess_image_as_array(image_file, show_output=True):
    im = Image.open(image_file).convert('RGB')
    im = im.resize((224, 224))

    return np.asarray(im)

def predict_image_class(model, img_array, class_names):
    img_batch = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_batch)
    predicted_class_index = np.argmax(predictions, axis=1)[0]

    if predicted_class_index < len(class_names):
        predicted_class = class_names[predicted_class_index]
        print(f"Predicted class: {predicted_class}")
        return predicted_class
    else:
        print(f"Predicted class index {predicted_class_index} out of range for class names")
        return None

@app.route('/predict', methods=['POST'])
# @firebase_authentication_middleware
def predict_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Image data not found'}), 400

        # Read image data from request
        image_data = request.files['image']

        # Convert FileStorage to a file-like object
        image_file = io.BytesIO(image_data.read())

        # Load and preprocess the image
        img_array = preprocess_image_as_array(image_file, show_output=False)

        # Predict the image
        predicted_class = predict_image_class(best_model, img_array, class_names)

        if predicted_class is not None:
            confidence = np.max(best_model.predict(np.expand_dims(img_array, axis=0))) * 100
            # save_prediction_to_firestore(predicted_class, confidence)
            return jsonify({'result': predicted_class, 'confidence': confidence}), 200
        else:
            return jsonify({'error': 'Failed to predict class for the image'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)