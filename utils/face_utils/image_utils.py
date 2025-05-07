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