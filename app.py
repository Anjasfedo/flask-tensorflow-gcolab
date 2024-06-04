import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import io
from model_load import model

# Set the environment variable to disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Image data not found'}), 400

        image_data = request.files['image'] # The uploaded image

        image_file = io.BytesIO(image_data.read()) # Process this image

        print(model.summary()) # Show model summary, indicate the model load successfuly
        
        # Handle image preprocessing same as ipynb does, then do predict and return it

        return jsonify({'result': 'lorem'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)