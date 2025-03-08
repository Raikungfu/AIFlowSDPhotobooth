import torch
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionDepth2ImgPipeline

torch.set_num_threads(4)

def dummy_checker(images, **kwargs):
    return images, [False] * len(images)

inpainting_pipeline = StableDiffusionInpaintPipeline.from_pretrained("botp/stable-diffusion-v1-5-inpainting", torch_dtype=torch.float16).to("cuda")

inpainting_pipeline.disable_attention_slicing()

inpainting_pipeline.safety_checker = dummy_checker