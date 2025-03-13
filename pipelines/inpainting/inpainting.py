import torch
from configs.model_paths import MODEL_PATHS
from diffusers import StableDiffusionControlNetInpaintPipeline

torch.set_num_threads(4)
torch.set_num_interop_threads(4)

def dummy_checker(images, **kwargs):
    return images, [False] * len(images)

inpainting_pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "botp/stable-diffusion-v1-5-inpainting",
    torch_dtype=torch.float16
).to("cuda")


ip_adapter = torch.load(MODEL_PATHS["ip_adapter"], map_location="cuda")
inpainting_pipeline.unet.load_state_dict(ip_adapter, strict=False)

inpainting_pipeline.disable_attention_slicing()
inpainting_pipeline.safety_checker = dummy_checker
