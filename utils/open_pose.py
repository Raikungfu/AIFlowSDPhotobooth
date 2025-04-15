from PIL import Image
import torch
from controlnet_aux import OpenposeDetector

def openpose_image(image):
    processor = OpenposeDetector.from_pretrained('lllyasviel/ControlNet',
            cache_dir="/content/pretrained")
    with torch.no_grad():
        processed_image_open_pose = processor(image, hand_and_face=True)
    
    return processed_image_open_pose