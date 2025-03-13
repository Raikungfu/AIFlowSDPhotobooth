from tkinter import Image


def create_canny_image(image):
    import cv2
    import numpy as np
    image = np.array(image.convert("L"))
    canny_image = cv2.Canny(image, 100, 200)
    canny_rgb = cv2.cvtColor(canny_image, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(canny_rgb)