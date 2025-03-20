import torch
from diffusers import ControlNetModel

def load_controlnet(model_name):
    return ControlNetModel.from_pretrained(
        model_name, torch_dtype=torch.float16, timeout=30
    )

# controlnet_openpose = load_controlnet("lllyasviel/control_v11p_sd15_openpose")
# controlnet_depth = load_controlnet("Shakker-Labs/FLUX.1-dev-ControlNet-Depth")
# controlnet_canny = load_controlnet("InstantX/FLUX.1-dev-Controlnet-Canny")