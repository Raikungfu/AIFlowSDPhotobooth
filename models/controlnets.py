import torch
from configs.model_paths import MODEL_PATHS
from diffusers import ControlNetModel

def load_controlnet(model_name):
    return ControlNetModel.from_pretrained(
        model_name, torch_dtype=torch.float16, timeout=30
    )

controlnet_openpose = load_controlnet(MODEL_PATHS["controlnet_openpose"])
controlnet_depth = load_controlnet(MODEL_PATHS["controlnet_depth"])
controlnet_canny = load_controlnet(MODEL_PATHS["controlnet_canny"])