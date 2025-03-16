from PIL import Image
import torch
from controlnet_aux import OpenposeDetector

def create_openpose_image(image):
    open_pose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
    with torch.no_grad():
        processed_image_open_pose = open_pose(image, hand_and_face=False)
    
    return processed_image_open_pose