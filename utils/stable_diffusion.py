import torch
from diffusers import StableDiffusionInpaintPipeline

torch.set_num_threads(4)

def dummy_checker(images, **kwargs):
    return images, [False] * len(images)

inpainting_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
    "sd-legacy/stable-diffusion-inpainting",
    revision="fp16",
    torch_dtype=torch.float16,
).to("cuda")

inpainting_pipeline.disable_attention_slicing()

inpainting_pipeline.safety_checker = dummy_checker