import torch, traceback
from configs.model_paths import MODEL_PATHS
from diffusers import (
    UniPCMultistepScheduler,
    StableDiffusionInpaintPipeline,
    StableDiffusionControlNetInpaintPipeline,
    EulerAncestralDiscreteScheduler
)
from models.controlnets import controlnet_openpose, controlnet_canny, controlnet_depth
from ip_adapter import IPAdapterPlus



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
            controlnet=selected_controlnets
        )

        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)

        pipeline.enable_attention_slicing()
        pipeline.enable_xformers_memory_efficient_attention()
        pipeline.safety_checker = dummy_checker
        
        pipeline.enable_model_cpu_offload()
        pipeline.to("cuda")

        pipeline.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="models",
            weight_name="ip-adapter-plus-face_sd15.bin",
            adapter_type=IPAdapterPlus,
        )
        
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
        )

        if pipeline is None:
            print("Inpainting_pipeline not loaded properly")
        else:
            print("Inpainting_pipeline loaded successfully")

        pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)

        pipeline.enable_attention_slicing()
        pipeline.enable_xformers_memory_efficient_attention()
        pipeline.safety_checker = dummy_checker
        
        pipeline.enable_model_cpu_offload()
        pipeline.to("cuda")
        
        print("Loaded Inpainting with Stable Diffusion 1.5 successfully")

        return pipeline
    except Exception as e:
        traceback.print_exc()
        print(f"Error during loading inpainting pipeline: {e}")
        return None
    


def load_ip_adapter(pipeline, ip_adapter_scale, num_images=1):
    pipeline.set_ip_adapter_scale([ [ip_adapter_scale for _ in range(num_images)] ])