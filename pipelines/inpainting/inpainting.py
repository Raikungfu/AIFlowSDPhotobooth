import traceback
import torch
from diffusers import (
    FluxControlNetModel, FluxMultiControlNetModel, FluxControlNetPipeline, UniPCMultistepScheduler
)


inpainting_pipeline = None

BASE_MODEL = "black-forest-labs/FLUX.1-dev"
CONTROLNET_UNION = "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro"
# LORA_PORTRAIT = "strangerzonehf/Flux-Super-Portrait-LoRA"


def load_inpainting_pipeline():
    global inpainting_pipeline
    if inpainting_pipeline is not None:
        return inpainting_pipeline
    
    try:
        def dummy_checker(images, **kwargs):
            return images, [False] * len(images)

        controlnet_union = FluxControlNetModel.from_pretrained(CONTROLNET_UNION, torch_dtype=torch.bfloat16)
        controlnet = FluxMultiControlNetModel([controlnet_union])

        inpainting_pipeline = FluxControlNetPipeline.from_pretrained(BASE_MODEL, controlnet=controlnet, torch_dtype=torch.bfloat16)
        inpainting_pipeline.to("cuda")

        # inpainting_pipeline.load_lora_weights(LORA_PORTRAIT)

        inpainting_pipeline.scheduler = UniPCMultistepScheduler.from_config(inpainting_pipeline.scheduler.config)

        inpainting_pipeline.safety_checker = dummy_checker
        inpainting_pipeline.enable_attention_slicing()
        inpainting_pipeline.enable_xformers_memory_efficient_attention()

        print("Loaded FLUX.1-dev with Union ControlNet")
        return inpainting_pipeline
    except Exception as e:
        traceback.print_exc()
        print(f"Error loading pipeline: {e}")
        return None

def get_pipeline():
    return inpainting_pipeline