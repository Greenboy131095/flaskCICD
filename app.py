from flask import Flask, jsonify
import firebase_admin
import os
from os import path
from PIL import Image
import numpy as np
from firebase_admin import credentials, firestore, storage
from sklearn.neighbors import KNeighborsClassifier
from urllib.parse import quote
import requests
from io import BytesIO

cred = credentials.Certificate('./serviceAccountKey.json')
firebase_admin.initialize_app(cred)

# Initialize Firestore DB
db = firestore.client()

# Initialize Firebase Storage
bucket = storage.bucket('instagram-7dc1d.appspot.com')

app = Flask(__name__)

def load_and_preprocess_image(image_path_or_url, target_size=(224, 224), dtype=np.float32, from_url=False):
    try:
        # Load the image from URL or file
        if from_url:
            response = requests.get(image_path_or_url)
            image = Image.open(BytesIO(response.content))
        else:
            image = Image.open(image_path_or_url)

        # Convert to RGB to ensure 3 channels (if not already)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize the image to the target size
        image = image.resize(target_size)

        # Convert the image to a NumPy array
        array = np.asarray(image, dtype=dtype)
        array=array.flatten()

        # Flatten the array if it's part of a dataset

        return array

    except Exception as e:
        print(f"Error loading and preprocessing image: {str(e)}")
        return None

def load_dataset(dataset_dir, target_size=(224, 224)):
    images = []
    labels = []
    label_to_index = {}  # Mapping from label names to numeric indices
    next_label_index = 0

    for root, dirs, files in os.walk(dataset_dir):
        for dir_name in dirs:
            label_to_index[dir_name] = next_label_index
            next_label_index += 1

        for dir_name in dirs:
            for file_name in os.listdir(os.path.join(root, dir_name)):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(root, dir_name, file_name)
                    label_index = label_to_index[dir_name]

                    image = load_and_preprocess_image(image_path, target_size=target_size, dtype=np.float32)
                    if image is not None:
                        images.append(image)
                        labels.append(label_index)

    # Create reverse mapping from index to label
    index_to_label = {v: k for k, v in label_to_index.items()}

    return images, labels, index_to_label


def build_knn_model(images, labels, n_neighbors=1):
    # Reshape the images array into (n_samples, n_features) for KNN
    images = np.array(images)
    if images.ndim == 4:
        # Assuming images are in shape (n_samples, height, width, channels)
        # We flatten the images
        n_samples, height, width, channels = images.shape
        images = images.reshape((n_samples, height * width * channels))
    
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(images, labels)
    return knn_model

# Load your own dataset and build a KNN model
current_dir = os.getcwd()
dataset_dir = path.join(current_dir, 'images')
images, labels , index_to_label = load_dataset(dataset_dir)
knn_model = build_knn_model(images, labels)

def classify_image(image_path):
    try:
        preprocessed_image = load_and_preprocess_image(image_path, from_url=True)
        predicted_label_index = knn_model.predict([preprocessed_image])[0]
        
        # Convert numeric label to actual label name
        predicted_label_name = index_to_label[predicted_label_index]
        
        return predicted_label_name

    except Exception as e:
        raise e

def get_latest_image_url(uid):
    try:
        # Construct the prefix based on the uid
        prefix = f"users/{uid}/uploads/"
        
        # List files in the specified directory using the imported bucket
        blobs = bucket.list_blobs(prefix=prefix)

        # Sort the blobs by upload time, and get the last one
        latest_blob = None
        for blob in sorted(blobs, key=lambda b: b.time_created, reverse=True):
            latest_blob = blob
            break  # Break after getting the latest blob

        if latest_blob:
            # Manually construct the Firebase Storage URL
            encoded_name = quote(latest_blob.name, safe='')
            # Return the public URL of the latest image
            return f"https://firebasestorage.googleapis.com/v0/b/{latest_blob.bucket.name}/o/{encoded_name}?alt=media"
        else:
            # Return None if no image found
            return None

    except Exception as e:
        # Handle exceptions if needed
        raise e

@app.route('/')
def hello_world():
    return 'Hello, Flask!'

@app.route('/imageClassification/<uid>', methods=['GET'])
def image_classification(uid):
    try:
        image_url = get_latest_image_url(uid)
        if image_url:
            category_name = classify_image(image_url)
            return jsonify({"category": category_name, "errorID": 0})
        else:
            return jsonify({"error": "No images found for the specified UID", "errorID": 1}), 404
    except Exception as e:
        return jsonify({"errorID": 3, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
