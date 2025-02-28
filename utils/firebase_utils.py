import firebase_admin
from firebase_admin import credentials, storage
import os

firebase_app = None

def initialize_firebase(cred_path, firebaseConfig):
    global firebase_app
    if not firebase_admin._apps:
        cred = credentials.Certificate(cred_path)
        firebase_app = firebase_admin.initialize_app(cred, {
            'storageBucket': firebaseConfig['storageBucket']
        })
    else:
        firebase_app = firebase_admin.get_app()
    
    return storage.bucket()

def upload_to_firebase_storage(file_path, folder_name, bucket):
    destination_blob_name = f"{folder_name}/{os.path.basename(file_path)}"
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(file_path)
    
    public_url = f"https://firebasestorage.googleapis.com/v0/b/{bucket.name}/o/{folder_name}%2F{os.path.basename(file_path)}?alt=media"
    print(f"Uploaded to Firebase: {public_url}")
    return public_url
