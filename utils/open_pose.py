from PIL import Image
import torch
from controlnet_aux import OpenposeDetector
from pipelines.inpainting import pipeline_instance


def load_openpose_processor():
    openpose_processor = OpenposeDetector.from_pretrained(
        'lllyasviel/ControlNet',
        cache_dir="/content/pretrained"
    )
    
    return openpose_processor


def openpose_image(image):
    with torch.inference_mode():
        processed_image_open_pose = pipeline_instance.openpose_processor(image, hand_and_face=True)
    
    return processed_image_open_pose

