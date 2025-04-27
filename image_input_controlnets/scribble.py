from PIL import Image
import cv2
import numpy as np

def scribble_image(image):
    try:
        if isinstance(image, np.ndarray):
            pass
        elif isinstance(image, Image.Image):
            image = np.array(image)
        else:
            raise ValueError("Input should be a PIL Image or numpy ndarray.")

        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        edges = cv2.Canny(image, threshold1=50, threshold2=150)

        scribble = 255 - edges

        scribble_image = Image.fromarray(scribble)

        return scribble_image
    except Exception as e:
        print(f"Error creating softedge image: {e}")
        import traceback
        traceback.print_exc()
        return None
