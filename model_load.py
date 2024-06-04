import os
import tensorflow as tf
# from google.cloud import storage 
from pathlib import Path

# Use os.path to construct the model path
model_dir = "model"
model_filename = "model_mm9.h5"
model_path = os.path.join(model_dir, model_filename)

# Load your model here
# cloud_storage_url = "https://storage.cloud.google.com/hijaiyah_model/model_mm9.h5"
# cloud_storage_url = "https://storage.cloud.google.com/hijaiyah_model/hijaiyah_final_model.h5"


# client = storage.Client.from_service_account_json(Path('serviceAccount.json').resolve())
# bucket = client.bucket("hijaiyah_model")
# blob = bucket.blob(model_filename)

# blob.download_to_filename(model_path)
# Load your model directly from the cloud storage
# best_model = None
# try:
best_model = tf.keras.models.load_model(model_path)
# except Exception as e:
#     best_model = None
#     print(f"Error = {e}")

# List of class names
class_names = [ 'ain', 'alif', 'ba', 'dal', 'dhod', 'dzal',
                'dzho', 'fa', 'ghoin', 'ha', 'ha\'', 'hamzah', 'jim',
                'kaf', 'kho', 'lam', 'lamalif', 'mim', 'nun', 'qof',
                'ro', 'shod', 'sin', 'syin', 'ta', 'tho', 'tsa',
                'wawu', 'ya', 'zain']