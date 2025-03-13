import torch
from diffusers import ControlNetModel

def load_controlnet(model_name):
    return ControlNetModel.from_pretrained(
        model_name, torch_dtype=torch.float16
    )

controlnet_openpose = load_controlnet("lllyasviel/control_v11p_sd15_openpose")
controlnet_depth = load_controlnet("lllyasviel/control_v11f1p_sd15_depth")
controlnet_canny = load_controlnet("lllyasviel/control_v11p_sd15_canny")