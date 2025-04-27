import cv2
import numpy as np
from PIL import Image

from utils.inpaint_pipeline_utils.masks import ensure_uint8

def canny_image(image):
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        image = image.convert("RGB").convert("L")
        image_np = np.array(image)
        image_np = ensure_uint8(image_np)

        canny_image = cv2.Canny(image_np, 100, 200)
        canny_rgb = cv2.cvtColor(canny_image, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(canny_rgb)
    except Exception as e:
        print(f"Error creating canny image: {e}")
        import traceback
        traceback.print_exc()
        return None
