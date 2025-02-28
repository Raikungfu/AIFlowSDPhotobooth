import firebase_admin
from firebase_admin import credentials, storage
import os

def initialize_firebase(cred_path, firebaseConfig):
    if not firebase_admin._apps:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred, {
            'storageBucket': firebaseConfig['storageBucket']
        })
    else:
        default_app = firebase_admin.get_app()

bucket = storage.bucket()

def upload_to_firebase_storage(file_path, folder_name):
    destination_blob_name = f"{folder_name}/{os.path.basename(file_path)}"
    blob = storage.bucket().blob(destination_blob_name)
    blob.upload_from_filename(file_path)
    public_url = f"https://firebasestorage.googleapis.com/v0/b/{firebaseConfig['storageBucket']}/o/{folder_name}%2F{os.path.basename(file_path)}?alt=media"
    print(f"Uploaded to Firebase: {public_url}")
    return public_url
