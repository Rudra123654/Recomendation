from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os
from PIL import Image
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Create upload folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load precomputed data
features_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load and configure model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    return result / norm(result)

def recommend(features, features_list):
    neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
    neighbors.fit(features_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

@app.route('/')
def home():
    return "Fashion Recommendation API is running!"

@app.route('/recommend', methods=['POST'])
def recommend_images():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(temp_path)
            
            features = feature_extraction(temp_path, model)
            indices = recommend(features, features_list)
            
            # Convert stored paths to serveable URLs
            recommendations = []
            for idx in indices[0]:
                img_path = filenames[idx]
                # Extract just the filename if full path is stored
                filename_only = os.path.basename(img_path)
                recommendations.append(filename_only)
            
            os.remove(temp_path)
            
            return jsonify({
                'status': 'success',
                'recommendations': recommendations
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Allowed file types are png, jpg, jpeg'}), 400

@app.route('/images/<filename>')
def serve_image(filename):
    # Try multiple possible locations for the image
    possible_locations = [
        '',  # Current directory
        'images',  # images folder
        'static/images',  # static/images folder
        os.path.dirname(filenames[0]) if filenames else ''  # First path in filenames
    ]
    
    for location in possible_locations:
        try:
            return send_from_directory(location, filename)
        except:
            continue
    
    return "Image not found", 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5500, debug=True)