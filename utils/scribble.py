import cv2
from PIL import Image

def scribble_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(image, threshold1=50, threshold2=150)

    scribble = 255 - edges

    scribble_image = Image.fromarray(scribble)

    return scribble_image