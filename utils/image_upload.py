from utils.firebase_utils import generate_random_filename, upload_to_firebase_storage
from PIL import Image
import requests
import io


def download_image(image_url):
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    except requests.RequestException as e:
        print(f"Error downloading image: {e}")
        return None
    

def upload_image(image, bucket):
    output_filename = generate_random_filename('.png')
    output_path = f'/tmp/{output_filename}'
    image.save(output_path)
    return upload_to_firebase_storage(output_path, 'processed_images', bucket)