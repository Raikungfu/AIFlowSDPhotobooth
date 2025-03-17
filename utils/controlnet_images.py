from PIL import Image
import numpy as np

from utils.canny import canny_image
from utils.depth import depth_image
from utils.open_pose import openpose_image

def process_image(image, processing_function):
    image = processing_function(image)
    return Image.fromarray(image) if isinstance(image, np.ndarray) else image

def create_depth_image(image):
    return process_image(image, depth_image)

def create_openpose_image(image):
    return process_image(image, openpose_image)

def create_canny_image(image):
    return process_image(image, canny_image)
