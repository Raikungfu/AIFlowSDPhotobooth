import torch
from configs.model_paths import MODEL_PATHS
from diffusers import ControlNetModel

def load_controlnet(model_name):
    return ControlNetModel.from_pretrained(
        model_name, torch_dtype=torch.float16, timeout=30
    )

controlnet_openpose = load_controlnet(MODEL_PATHS["openpose_model"])
controlnet_depth = load_controlnet(MODEL_PATHS["depth_model"])
controlnet_canny = load_controlnet(MODEL_PATHS["canny_model"])