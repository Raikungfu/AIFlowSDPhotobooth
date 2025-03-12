import torch
from diffusers import StableDiffusionInpaintPipeline, ControlNetModel

torch.set_num_threads(4)

def dummy_checker(images, **kwargs):
    return images, [False] * len(images)

controlnet_pipeline = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_openpose",
    torch_dtype=torch.float16
).to("cuda")


inpainting_pipeline = StableDiffusionInpaintPipeline.from_pretrained("botp/stable-diffusion-v1-5-inpainting",
    controlnet=controlnet_pipeline,
    torch_dtype=torch.float16).to("cuda")

# ip_adapter_weights = torch.load("models/ip-adapter-plus-face_sd15.pth", map_location="cuda")

# inpainting_pipeline.unet.load_attn_procs(ip_adapter_weights)

inpainting_pipeline.disable_attention_slicing()

inpainting_pipeline.safety_checker = dummy_checker