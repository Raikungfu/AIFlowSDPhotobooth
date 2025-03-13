import torch
from diffusers import StableDiffusionInpaintPipeline, ControlNetModel

torch.set_num_threads(4)

def dummy_checker(images, **kwargs):
    return images, [False] * len(images)

controlnet_openpose = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16
).to("cuda")

controlnet_depth = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16
).to("cuda")

controlnet_canny = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16
).to("cuda")

controlnets = [controlnet_openpose, controlnet_depth, controlnet_canny]

inpainting_pipeline = StableDiffusionInpaintPipeline.from_pretrained("botp/stable-diffusion-v1-5-inpainting",
    controlnet=controlnets,
    torch_dtype=torch.float16).to("cuda")

# ip_adapter_weights = torch.load("models/ip-adapter-plus-face_sd15.pth", map_location="cuda")

# inpainting_pipeline.unet.load_attn_procs(ip_adapter_weights)

inpainting_pipeline.disable_attention_slicing()

inpainting_pipeline.safety_checker = dummy_checker