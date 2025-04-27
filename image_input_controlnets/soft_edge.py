from PIL import Image
import cv2
import numpy as np

def softedge_image(image):
    try:
        if isinstance(image, np.ndarray):
            pass
        elif isinstance(image, Image.Image):
            image = np.array(image)
        else:
            raise ValueError("Input should be a PIL Image or numpy ndarray.")

        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

        sobel_combined = cv2.magnitude(sobelx, sobely)

        softedge = np.uint8(255 * (sobel_combined / np.max(sobel_combined)))

        softedge_image = Image.fromarray(softedge)

        return softedge_image
    except Exception as e:
        print(f"Error creating softedge image: {e}")
        import traceback
        traceback.print_exc()
        return None
