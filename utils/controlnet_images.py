from PIL import Image
import numpy as np
from utils.canny import create_canny_image
from utils.depth import create_depth_image
from utils.open_pose import create_openpose_image

def create_controlnet_image(image):
    depth_image = create_depth_image(image)
    if isinstance(depth_image, np.ndarray):
        depth_image = Image.fromarray(depth_image)

    openpose_image = create_openpose_image(image)
    if isinstance(openpose_image, np.ndarray):
        openpose_image = Image.fromarray(openpose_image)

    canny_image = create_canny_image(image)

    if isinstance(canny_image, np.ndarray):
        canny_image = Image.fromarray(canny_image)
    
    return depth_image, openpose_image, canny_image
