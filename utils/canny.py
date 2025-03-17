import cv2
import numpy as np
from PIL import Image

def canny_image(image):
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        image = image.convert("L")
        image_np = np.array(image)
        if image_np.dtype != np.uint8:
            image_np = image_np.astype(np.uint8)

        canny_image = cv2.Canny(image_np, 100, 200)
        canny_rgb = cv2.cvtColor(canny_image, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(canny_rgb)
    except Exception as e:
        print(f"Error creating canny image: {e}")
        import traceback
        traceback.print_exc()
        return None
