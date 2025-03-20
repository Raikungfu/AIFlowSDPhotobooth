import traceback
import torch
from diffusers import (
    FluxControlNetModel, FluxMultiControlNetModel, FluxControlNetPipeline, UniPCMultistepScheduler
)

BASE_MODEL = "black-forest-labs/FLUX.1-dev"
CONTROLNET_UNION = "InstantX/FLUX.1-dev-Controlnet-Union"

LORA_PORTRAIT = "strangerzonehf/Flux-Super-Portrait-LoRA"

def load_inpainting_pipeline():
    try:
        # def dummy_checker(images, **kwargs):
        #     return images, [False] * len(images)

        controlnet_union = FluxControlNetModel.from_pretrained(
            CONTROLNET_UNION, torch_dtype=torch.bfloat16
        ).to("cuda")

        controlnet = FluxMultiControlNetModel([controlnet_union])

        pipeline = FluxControlNetPipeline.from_pretrained(
            BASE_MODEL, controlnet=controlnet, torch_dtype=torch.bfloat16
        ).to("cuda")

        pipeline.load_lora_weights(LORA_PORTRAIT)

        pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
        
        pipeline.enable_xformers_memory_efficient_attention()

        pipeline.safety_checker = None 

        pipeline.enable_attention_slicing()

        print("Loaded FLUX.1-dev with Union ControlNet")
        return pipeline
    except Exception as e:
        traceback.print_exc()
        print(f"Error loading pipeline: {e}")
        return None
    
inpainting_pipeline = load_inpainting_pipeline()
