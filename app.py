from flask import Flask
from flask import jsonify
import firebase_admin
from firebase_admin import credentials, firestore,storage
from urllib.parse import quote # import the quote function 
cred = credentials.Certificate('./serviceAccountKey.json')
firebase_admin.initialize_app(cred)
# Initialize Firestore DB
db = firestore.client()

# Initialize Firebase Storage
bucket = storage.bucket('instagram-7dc1d.appspot.com')
app = Flask(__name__)
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
        # Get the latest image URL
        image_url = get_latest_image_url(uid)

        if image_url:
            return jsonify({"category": image_url,"errorID":0})
        else:
            return jsonify({"error": "No images found for the specified UID","errorID":1}), 404

    except Exception as e:
        return jsonify({"errorID":3, "error": str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True) 