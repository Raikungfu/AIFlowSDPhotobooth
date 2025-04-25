import torch, traceback
from configs.model_paths import MODEL_PATHS
from diffusers import (
    UniPCMultistepScheduler,
    StableDiffusionInpaintPipeline,
    StableDiffusionControlNetInpaintPipeline
)
from models.controlnets import controlnet_openpose, controlnet_canny, controlnet_depth

def dummy_checker(images, **kwargs):
    return images, [False] * len(images)

def load_inpainting_controlnet_pipeline(controlnets=None):
    try:
        controlnet_map = {
            "pose": controlnet_openpose,
            "canny": controlnet_canny,
            "depth": controlnet_depth
        }

        selected_controlnets = [controlnet_map[name] for name in controlnets] if controlnets else None
            
        pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            MODEL_PATHS["inpainting_model"],
            torch_dtype=torch.float16,
            # low_cpu_mem_usage=True,
            controlnet=selected_controlnets
        )

        pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)

        ip_adapter = torch.load(MODEL_PATHS["ip_adapter"], map_location="cuda")
        pipeline.unet.load_state_dict(ip_adapter, strict=False)

        pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)

        pipeline.enable_attention_slicing()
        pipeline.enable_xformers_memory_efficient_attention()
        pipeline.safety_checker = dummy_checker
        
        # pipeline.enable_model_cpu_offload()
        pipeline.to("cuda")
        
        print(f"Loaded Inpainting with Stable Diffusion 1.5 successfully with controlnets: {controlnets}")

        return pipeline
    except Exception as e:
        print(f"Error during loading inpainting pipeline: {e}")
        return None


def load_inpainting_pipeline():
    try:
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            MODEL_PATHS["inpainting_model"],
            torch_dtype=torch.float16,
            # low_cpu_mem_usage=True,
        ).to("cuda")

        if pipeline is None:
            print("Inpainting_pipeline not loaded properly")
        else:
            print("Inpainting_pipeline loaded successfully")

        ip_adapter = torch.load(MODEL_PATHS["ip_adapter"])
        pipeline.unet.load_state_dict(ip_adapter, strict=False)

        pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)

        pipeline.enable_attention_slicing()
        pipeline.enable_xformers_memory_efficient_attention()
        pipeline.safety_checker = dummy_checker
        
        # pipeline.enable_model_cpu_offload()
        
        print("Loaded Inpainting with Stable Diffusion 1.5 successfully")

        return pipeline
    except Exception as e:
        traceback.print_exc()
        print(f"Error during loading inpainting pipeline: {e}")
        return None