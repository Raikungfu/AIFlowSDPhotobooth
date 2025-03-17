from PIL import Image
import numpy as np
from utils.canny import canny_image
from utils.depth import depth_image
from utils.open_pose import openpose_image

def create_depth_image(image):
    depth_image = depth_image(image)
    if isinstance(depth_image, np.ndarray):
        depth_image = Image.fromarray(depth_image)

    return depth_image


def create_openpose_image(image):
    openpose_image = openpose_image(image)
    if isinstance(openpose_image, np.ndarray):
        openpose_image = Image.fromarray(openpose_image)
    
    return openpose_image


def create_canny_image(image):
    canny_image = canny_image(image)

    if isinstance(canny_image, np.ndarray):
        canny_image = Image.fromarray(canny_image)
    
    return canny_image
