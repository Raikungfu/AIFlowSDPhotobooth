from PIL import Image
import cv2
import numpy as np


def softedge_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    sobel_combined = cv2.magnitude(sobelx, sobely)

    softedge = np.uint8(255 * (sobel_combined / np.max(sobel_combined)))

    softedge_image = Image.fromarray(softedge)

    return softedge_image