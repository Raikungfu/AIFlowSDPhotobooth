import base64
import io
from PIL import Image
import numpy as np


def convert_image_to_numpy(image):
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    return image


def ensure_uint8(image_np):
    if image_np.dtype != np.uint8:
        return image_np.astype(np.uint8)
    return image_np


def ensure_pil_images(images):
    pil_images = []
    for img in images:
        if isinstance(img, Image.Image):
            pil_images.append(img)
        elif isinstance(img, str) and img.startswith("data:image"):
            img = Image.open(io.BytesIO(base64.b64decode(img.split(',')[1]))).convert("RGB")
            pil_images.append(img)
        else:
            raise ValueError(f"Unsupported image format in face_images: {type(img)}")
    return pil_images